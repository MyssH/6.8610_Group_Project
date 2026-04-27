from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    dtype: torch.dtype

    @property
    def dtype_name(self) -> str:
        return str(self.dtype).replace("torch.", "")


def choose_device(prefer_mps: bool = True) -> DeviceInfo:
    if prefer_mps and torch.backends.mps.is_available():
        return DeviceInfo(torch.device("mps"), torch.float16)
    if torch.cuda.is_available():
        return DeviceInfo(torch.device("cuda"), torch.float16)
    return DeviceInfo(torch.device("cpu"), torch.float32)
