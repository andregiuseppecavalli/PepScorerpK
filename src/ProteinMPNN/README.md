# ProteinMPNN backbone

Copy these from the official ProteinMPNN repository
(https://github.com/dauparas/ProteinMPNN):

    src/ProteinMPNN/protein_mpnn_utils.py
    src/ProteinMPNN/vanilla_model_weights/v_48_020.pt

`protein_mpnn_utils.py` provides the `ProteinMPNN` class used by
`proteinMPNN_model_score.py`. `v_48_020.pt` is the k=48 vanilla checkpoint.
No `__init__.py` is required (namespace package).
