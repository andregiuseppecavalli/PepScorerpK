import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ProteinMPNN.protein_mpnn_utils import ProteinMPNN
except ImportError:
    raise ImportError(
        "File 'protein_mpnn_utils.py' not found. Please copy it from the "
        "ProteinMPNN repository into src/ProteinMPNN/."
    )

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

def gather_nodes(nodes, neighbor_idx):
    """Gather node features at neighbor indices.
    nodes:        [B, N, C]
    neighbor_idx: [B, N, K]
    returns:      [B, N, K, C]
    """
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    nodes_neighbors = torch.gather(
        nodes, 1,
        neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.shape[2]),
    )
    return nodes_neighbors.view(
        (neighbor_idx.shape[0], neighbor_idx.shape[1], -1, nodes.shape[2])
    )


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """For each (i, neighbor j of i), concatenate h_neighbors[i,j] with h_nodes[j]."""
    h_nodes_gathered = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes_gathered], -1)

class ProteinMPNNEncoderScorer(nn.Module):
    """
    Wraps a frozen ProteinMPNN and returns BOTH:

      * Structural embeddings (encoder pooling) -> [B, 512]
      * Peptide sequence-scoring features       -> [B, 24]

    Score features for the peptide (24 dims):
        0     pep_mean_logprob   mean log P(true_aa) over peptide positions
        1     pep_recovery       fraction of positions with argmax == true
        2     pep_mean_entropy   mean predictive entropy at peptide positions
        3..23 pep_pred_<AA>      mean predicted distribution over peptide
                                 positions (21 dims)
    """

    EMB_DIM = 512    
    SCORE_DIM = 24  

    def __init__(self, checkpoint_path, device, k_neighbors=48):
        super().__init__()
        self.mpnn = ProteinMPNN(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0,
            k_neighbors=k_neighbors,
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.mpnn.load_state_dict(ckpt["model_state_dict"])
        self.mpnn.to(device)
        self.mpnn.eval()

    @staticmethod
    def score_feature_names():
        names = ["pep_mean_logprob", "pep_recovery", "pep_mean_entropy"]
        names += [f"pep_pred_{aa}" for aa in MPNN_ALPHABET]
        return names

    def forward(self, X, mask, chain_enc, residue_idx, S, m_pep, m_pock):
        """
        Args:
            X           [B, L, 4, 3]   N, CA, C, O coordinates
            mask        [B, L]         1 valid, 0 padding
            chain_enc   [B, L]         integer chain encoding (1, 2, ...)
            residue_idx [B, L]         integer residue indices
            S           [B, L]         AA indices in the ProteinMPNN alphabet
            m_pep       [B, L]         1 for peptide residues
            m_pock      [B, L]         1 for pocket residues

        Returns dict: {'embeddings': [B, 512], 'score_features': [B, 24]}
        """
        with torch.no_grad():
            B, L = mask.shape
            device = X.device

            E, E_idx = self.mpnn.features(X, mask, residue_idx, chain_enc)
            h_V = torch.zeros((B, L, E.shape[-1]), device=device)
            h_E = self.mpnn.W_e(E)

            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            for layer in self.mpnn.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

            h_V_enc = h_V

            pep_denom  = torch.clamp(m_pep.sum(1, keepdim=True),  min=1e-9)
            pock_denom = torch.clamp(m_pock.sum(1, keepdim=True), min=1e-9)

            pep_vec  = (h_V_enc * m_pep.unsqueeze(-1)).sum(1)  / pep_denom
            pock_vec = (h_V_enc * m_pock.unsqueeze(-1)).sum(1) / pock_denom
            prod = pep_vec * pock_vec
            diff = torch.abs(pep_vec - pock_vec)
            emb_feat = torch.cat([pep_vec, pock_vec, prod, diff], dim=1)

            chain_M_design = m_pep * mask

            randn = torch.randn(chain_M_design.shape, device=device)
            decoding_order = torch.argsort(
                (chain_M_design + 0.0001) * torch.abs(randn)
            )

            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = F.one_hot(
                decoding_order, num_classes=mask_size
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend_dec = order_mask_backward.gather(2, E_idx).unsqueeze(-1)
            mask_1D = mask.view(B, L, 1, 1)
            mask_bw = mask_1D * mask_attend_dec
            mask_fw = mask_1D * (1.0 - mask_attend_dec)

            h_S = self.mpnn.W_s(S)
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            h = h_V_enc
            for layer in self.mpnn.decoder_layers:
                h_ESV = cat_neighbors_nodes(h, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h = layer(h, h_ESV, mask)

            logits = self.mpnn.W_out(h)               
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            true_logprob_per_res = torch.gather(
                log_probs, 2, S.unsqueeze(-1)
            ).squeeze(-1)
            pep_mean_logprob = (
                (true_logprob_per_res * m_pep).sum(1, keepdim=True) / pep_denom
            )

            argmax_S = log_probs.argmax(dim=-1)
            recovery = (
                ((argmax_S == S).float() * m_pep).sum(1, keepdim=True) / pep_denom
            )

            entropy_per_res = -(probs * log_probs).sum(-1)
            pep_mean_entropy = (
                (entropy_per_res * m_pep).sum(1, keepdim=True) / pep_denom
            )

            pep_mean_dist = (probs * m_pep.unsqueeze(-1)).sum(1) / pep_denom

            score_feat = torch.cat(
                [pep_mean_logprob, recovery, pep_mean_entropy, pep_mean_dist],
                dim=1,
            ) 

        return {"embeddings": emb_feat, "score_features": score_feat}
