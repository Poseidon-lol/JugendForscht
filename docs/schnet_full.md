# Full SchNet Surrogate (3D)

This project now includes a **real SchNet** surrogate (PyG `torch_geometric.nn.models.SchNet`) alongside the lightweight pseudo-SchNet.

## Training
```bash
python src/main.py train-surrogate-3d-full --config configs/train_conf_3d_full.yaml
```
- Expects MolBlocks in `mol` (default) and optional SMILES in `smile`/`smiles`.
- Targets come from `dataset.target_columns` in the config.
- Checkpoints are saved to `training.save_dir` (default `models/surrogate_3d_full/schnet_full.pt`).

## Dependencies
- Requires `torch_geometric` and `torch_scatter` (SchNet CFConv kernels).
- DirectML is **not** supported; the loader falls back to CPU if DirectML is requested.

## Active-Learning Use
- Point `--surrogate-dir` to the SchNet checkpoint (file `schnet_full.pt` or a directory containing it).
- If present, the AL loop loads this model; otherwise it falls back to pseudo-SchNet or the MPNN ensemble.
