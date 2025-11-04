# AMD GPU Training Guide

This project now auto-detects the best torch backend across CUDA, Apple MPS,
DirectML (Windows + AMD) and CPU via `src/utils/device.py`.  Follow the steps
below to run the training stack on an AMD GPU.

## 1. Install a GPU-enabled PyTorch build

- **Windows + AMD**: install the DirectML build
  ```powershell
  pip install torch-directml
  ```
  This works with the official CPU wheels of PyTorch ≥ 2.1.  Keep using the
  regular `torch` package; `torch_directml` is loaded dynamically.

- **Linux + ROCm**: install the ROCm wheel that matches your driver.  Example:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
  ```

After installation validate the backend:
```python
python - <<'PY'
import torch
try:
    import torch_directml
    print("DirectML available:", torch_directml.is_available())
except ImportError:
    print("torch_directml not installed")
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
PY
```

## 2. Training entry points

All high-level helpers (`train_ensemble`, `ensemble_predict`, `train_jtvae`,
and the notebook utilities) now call `get_device()` which chooses:
`cuda → mps → directml → cpu`.  When `torch-directml` is installed, your AMD
GPU is picked automatically.

If you need to override the selection, pass an explicit device identifier:
```python
from src.models.mpnn import train_ensemble

# Force DirectML
train_ensemble(df, "models/surrogate", device="directml")

# Explicit GPU index on ROCm/CUDA
train_ensemble(df, "models/surrogate", device="cuda:0")
```

The same parameter is available for:
- `src/models/ensemble.py::train_ensemble` and `ensemble_predict`
- `src/models/jtvae_extended.py::train_jtvae`
- Notebook cells that call these functions (pass `device="directml"`).

## 3. Known limitations

- PyTorch Geometric wheels currently target CUDA.  For DirectML you may need to
  build PyG from source or rely on CPU fallback operators if a kernel is
  missing.
- Mixed precision (`torch.cuda.amp`) now kicks in automatically on CUDA builds
  (configurable via `use_amp` / `--amp`).  DirectML/ROCm still run in FP32
  because AMP kernels are not available there yet.
- Loading checkpoints is always mapped through CPU for DirectML to ensure the
  runtime stays stable; this has a small overhead compared to CUDA.

## 4. Troubleshooting

- If PyTorch reports `RuntimeError: No kernel for op …` on DirectML, fall back
  to CPU for that stage by setting `device="cpu"` temporarily.
- Update to the latest AMD drivers when using DirectML to avoid crashes in
  long training runs.

With these steps you can run the surrogate and JT-VAE training loops on AMD
GPUs without code changes to your training scripts.
