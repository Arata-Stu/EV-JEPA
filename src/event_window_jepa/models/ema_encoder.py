from __future__ import annotations

import copy

import torch
from torch import nn


def make_ema_copy(module: nn.Module) -> nn.Module:
    target = copy.deepcopy(module)
    target.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    if not 0 <= momentum <= 1:
        raise ValueError("momentum must lie in [0, 1]")
    online_parameters = dict(online.named_parameters())
    target_parameters = dict(target.named_parameters())
    if online_parameters.keys() != target_parameters.keys():
        raise ValueError("online and target encoders have different parameters")
    for name, target_parameter in target_parameters.items():
        online_parameter = online_parameters[name]
        target_parameter.mul_(momentum).add_(online_parameter.detach(), alpha=1.0 - momentum)

    online_buffers = dict(online.named_buffers())
    target_buffers = dict(target.named_buffers())
    if online_buffers.keys() != target_buffers.keys():
        raise ValueError("online and target encoders have different buffers")
    for name, target_buffer in target_buffers.items():
        target_buffer.copy_(online_buffers[name])

