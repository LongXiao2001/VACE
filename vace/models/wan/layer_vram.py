import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def cast_to(tensor, dtype, device):
    result = torch.empty_like(tensor, dtype=dtype, device=device)
    result.copy_(tensor)
    return result


class AutoWrappedModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        offload_dtype,
        offload_device,
        onload_dtype,
        onload_device,
        computation_dtype,
        computation_device,
        verbose=False,
    ):
        super().__init__()
        self.module = module.to(dtype=offload_dtype, device=offload_device)
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.verbose = verbose
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0
            if self.verbose:
                logging.info("[LayerVRAM] offload %s -> %s", self.__class__.__name__, self.offload_device)

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1
            if self.verbose:
                logging.info("[LayerVRAM] onload %s -> %s", self.__class__.__name__, self.onload_device)

    def forward(self, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            module = self.module
        else:
            module = copy.deepcopy(self.module).to(dtype=self.computation_dtype, device=self.computation_device)
        return module(*args, **kwargs)


class AutoWrappedLinear(nn.Module):
    def __init__(
        self,
        module: nn.Linear,
        offload_dtype,
        offload_device,
        onload_dtype,
        onload_device,
        computation_dtype,
        computation_device,
        verbose=False,
    ):
        super().__init__()
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.weight = nn.Parameter(module.weight.detach().to(dtype=offload_dtype, device=offload_device), requires_grad=False)
        self.bias = None if module.bias is None else nn.Parameter(module.bias.detach().to(dtype=offload_dtype, device=offload_device), requires_grad=False)
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.verbose = verbose
        self.state = 0

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0
            if self.verbose:
                logging.info("[LayerVRAM] offload Linear -> %s", self.offload_device)

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1
            if self.verbose:
                logging.info("[LayerVRAM] onload Linear -> %s", self.onload_device)

    def forward(self, x, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            weight, bias = self.weight, self.bias
        else:
            weight = cast_to(self.weight, self.computation_dtype, self.computation_device)
            bias = None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)
        return F.linear(x, weight, bias)


def enable_layer_vram_recursively(
    model: nn.Module,
    module_map: dict,
    module_config: dict,
    max_num_param=None,
    overflow_module_config=None,
    total_num_param=0,
):
    for name, module in model.named_children():
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                num_param = sum(p.numel() for p in module.parameters())
                if max_num_param is not None and total_num_param + num_param > max_num_param:
                    current_config = overflow_module_config
                else:
                    current_config = module_config
                wrapped = target_module(module, **current_config)
                setattr(model, name, wrapped)
                total_num_param += num_param
                break
        else:
            total_num_param = enable_layer_vram_recursively(
                module,
                module_map,
                module_config,
                max_num_param=max_num_param,
                overflow_module_config=overflow_module_config,
                total_num_param=total_num_param,
            )
    return total_num_param


def enable_layer_vram_management(
    model: nn.Module,
    module_map: dict,
    module_config: dict,
    max_num_param=None,
    overflow_module_config=None,
):
    enable_layer_vram_recursively(
        model,
        module_map,
        module_config,
        max_num_param=max_num_param,
        overflow_module_config=overflow_module_config,
        total_num_param=0,
    )
    model.layer_vram_management_enabled = True
    model.layer_vram_onload_device = module_config["onload_device"]
    model.layer_vram_offload_device = module_config["offload_device"]


def wrapped_model_onload(model: nn.Module):
    if hasattr(model, "layer_vram_management_enabled") and model.layer_vram_management_enabled:
        for module in model.modules():
            if not hasattr(module, "onload"):
                for name, param in list(module._parameters.items()):
                    if param is not None:
                        module._parameters[name] = nn.Parameter(
                            param.to(device=model.layer_vram_onload_device),
                            requires_grad=param.requires_grad,
                        )
                for name, buffer in list(module._buffers.items()):
                    if buffer is not None:
                        module._buffers[name] = buffer.to(device=model.layer_vram_onload_device)
            if hasattr(module, "onload"):
                module.onload()


def wrapped_model_offload(model: nn.Module):
    if hasattr(model, "layer_vram_management_enabled") and model.layer_vram_management_enabled:
        for module in model.modules():
            if not hasattr(module, "offload"):
                for name, param in list(module._parameters.items()):
                    if param is not None:
                        module._parameters[name] = nn.Parameter(
                            param.to(device=model.layer_vram_offload_device),
                            requires_grad=param.requires_grad,
                        )
                for name, buffer in list(module._buffers.items()):
                    if buffer is not None:
                        module._buffers[name] = buffer.to(device=model.layer_vram_offload_device)
            if hasattr(module, "offload"):
                module.offload()
