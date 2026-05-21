import os
from contextlib import contextmanager

import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:  # pragma: no cover
    safe_load_file = None


def _load_state_dict(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        if safe_load_file is None:
            raise ImportError("safetensors is required to load .safetensors LoRA weights.")
        return safe_load_file(path, device="cpu")
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


class LoRAManager:
    def __init__(self, model, lora_path, scale=1.0, dynamic_offload=True):
        self.model = model
        self.lora_path = lora_path
        self.scale = scale
        self.dynamic_offload = dynamic_offload
        self.active = False
        self.deltas = self._build_deltas()

    def _build_deltas(self):
        state_dict = _load_state_dict(self.lora_path)
        modules = dict(self.model.named_modules())
        grouped = {}

        for key, value in state_dict.items():
            if key.endswith(".lora_A.weight"):
                base = key[:-len(".lora_A.weight")]
                grouped.setdefault(base, {})["a"] = value.float()
            elif key.endswith(".lora_B.weight"):
                base = key[:-len(".lora_B.weight")]
                grouped.setdefault(base, {})["b"] = value.float()
            elif key.endswith(".lora_down.weight"):
                base = key[:-len(".lora_down.weight")]
                grouped.setdefault(base, {})["a"] = value.float()
            elif key.endswith(".lora_up.weight"):
                base = key[:-len(".lora_up.weight")]
                grouped.setdefault(base, {})["b"] = value.float()
            elif key.endswith(".alpha"):
                base = key[:-len(".alpha")]
                grouped.setdefault(base, {})["alpha"] = float(value)

        deltas = {}
        for module_name, parts in grouped.items():
            if "a" not in parts or "b" not in parts:
                continue
            module = modules.get(module_name)
            if module is None or not hasattr(module, "weight"):
                continue

            down_weight = parts["a"]
            up_weight = parts["b"]
            rank = max(1, down_weight.shape[0])
            alpha = parts.get("alpha", rank)
            lora_scale = self.scale * (alpha / rank)

            if isinstance(module, nn.Linear):
                delta = torch.matmul(up_weight, down_weight)
                delta = delta.view_as(module.weight).contiguous()
            elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
                delta = torch.matmul(
                    up_weight.reshape(up_weight.shape[0], -1),
                    down_weight.reshape(down_weight.shape[0], -1),
                )
                delta = delta.view_as(module.weight).contiguous()
            else:
                continue

            deltas[module_name] = delta.mul_(lora_scale).cpu()
        return deltas

    def apply(self):
        if self.active:
            return
        modules = dict(self.model.named_modules())
        for module_name, delta in self.deltas.items():
            module = modules[module_name]
            delta_device = delta.to(device=module.weight.device, dtype=module.weight.dtype)
            module.weight.data.add_(delta_device)
        self.active = True

    def remove(self):
        if not self.active:
            return
        modules = dict(self.model.named_modules())
        for module_name, delta in self.deltas.items():
            module = modules[module_name]
            delta_device = delta.to(device=module.weight.device, dtype=module.weight.dtype)
            module.weight.data.sub_(delta_device)
        self.active = False

    @contextmanager
    def activated(self):
        self.apply()
        try:
            yield
        finally:
            if self.dynamic_offload:
                self.remove()
