from __future__ import annotations

import torch
from torch import nn

from event_window_jepa.models.ema_encoder import update_ema


def test_ema_update_matches_definition() -> None:
    online = nn.Linear(2, 1, bias=False)
    target = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        online.weight.fill_(2.0)
        target.weight.fill_(0.0)
    update_ema(online, target, momentum=0.75)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.5))

