import os
import pytest


def _try_import():
    try:
        import torch  # noqa: F401
        from torch_geometric.data import Data  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _try_import(), reason="torch_geometric not available")


def test_schnet_full_forward_minimal():
    import torch
    from torch_geometric.data import Data
    from src.models.schnet_full import RealSchNetModel, RealSchNetConfig

    z = torch.tensor([6, 6, 8], dtype=torch.long)  # C C O
    pos = torch.randn(3, 3)
    batch = torch.zeros(3, dtype=torch.long)
    y = torch.tensor([[0.0, 1.0]])  # two targets
    data = Data(z=z, pos=pos, y=y, batch=batch)

    model = RealSchNetModel(RealSchNetConfig(hidden_channels=32, num_filters=32, num_interactions=2, num_gaussians=10), out_dim=2)
    out = model(data.z, data.pos, data.batch)
    assert out.shape == (1, 2)
