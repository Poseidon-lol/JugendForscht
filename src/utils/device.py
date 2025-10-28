"""Utility helpers to select the best available torch device across backends.

This module consolidates the logic for working with NVIDIA CUDA, Apple MPS,
AMD/Windows DirectML (torch-directml), and CPU execution.  It returns a thin
DeviceSpec wrapper so the rest of the codebase can stay backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

try:
    import torch_directml  # type: ignore

    _DIRECTML_AVAILABLE = True
except ImportError:
    torch_directml = None  # type: ignore
    _DIRECTML_AVAILABLE = False


@dataclass(frozen=True)
class DeviceSpec:
    """Represents a resolved device handle and metadata."""

    target: Any
    type: str
    index: Optional[int] = None

    @property
    def map_location(self) -> Union[str, torch.device]:
        """Value usable for torch.load(map_location=...)."""
        return "cpu" if self.type == "directml" else self.target

    def as_torch_device(self) -> Optional[torch.device]:
        return self.target if isinstance(self.target, torch.device) else None


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def get_device(device: Optional[Union[str, torch.device, DeviceSpec]] = None) -> DeviceSpec:
    """Resolve a DeviceSpec from user input or auto-detection."""
    if isinstance(device, DeviceSpec):
        return device
    if isinstance(device, torch.device):
        return DeviceSpec(target=device, type=device.type, index=device.index)
    if isinstance(device, str):
        lowered = device.strip().lower()
        if lowered.startswith("cuda") or lowered.startswith("hip"):
            if not torch.cuda.is_available():
                raise ValueError("CUDA/HIP requested but no compatible GPU is available.")
            dev = torch.device(lowered if lowered != "hip" else "cuda")
            return DeviceSpec(target=dev, type="cuda", index=dev.index)
        if lowered.startswith("mps"):
            if not _mps_available():
                raise ValueError("MPS requested but it is not available on this system.")
            dev = torch.device("mps")
            return DeviceSpec(target=dev, type="mps", index=dev.index)
        if lowered in {"directml", "dml"}:
            if not _DIRECTML_AVAILABLE:
                raise ValueError("DirectML requested but torch-directml is not installed.")
            return DeviceSpec(target=torch_directml.device(), type="directml", index=0)  # type: ignore[arg-type]
        if lowered == "cpu":
            return DeviceSpec(target=torch.device("cpu"), type="cpu", index=None)
        raise ValueError(f"Unrecognised device specifier '{device}'.")

    # Auto-detect best available backend
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        return DeviceSpec(target=dev, type="cuda", index=dev.index)
    if _mps_available():
        dev = torch.device("mps")
        return DeviceSpec(target=dev, type="mps", index=dev.index)
    if _DIRECTML_AVAILABLE:
        return DeviceSpec(target=torch_directml.device(), type="directml", index=0)  # type: ignore[arg-type]
    return DeviceSpec(target=torch.device("cpu"), type="cpu", index=None)


def move_to_device(obj: Any, device: DeviceSpec) -> Any:
    """Move a tensor/module/Data object to the selected device."""
    return obj.to(device.target)


def ensure_state_dict_on_cpu(model: torch.nn.Module, device: DeviceSpec) -> dict[str, torch.Tensor]:
    """Return a state dict that is safe to serialise regardless of backend."""
    state = model.state_dict()
    if device.type == "directml":
        return {k: v.detach().cpu() for k, v in state.items()}
    return state


def is_directml(device: DeviceSpec) -> bool:
    return device.type == "directml"
