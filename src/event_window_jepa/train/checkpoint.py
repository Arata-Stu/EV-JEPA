from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.models.window_jepa import WindowJEPA


SCHEMA_VERSION = 2


def config_hash(config: ExperimentConfig) -> str:
    resolved = config.to_dict()
    # Keep schema-v2 checkpoints created before the optional V-JEPA 2.1
    # architecture fields loadable. Their implicit architecture is v1.
    model = resolved["model"]
    if model.get("architecture") == "event_vit_v1":
        model.pop("architecture", None)
        if not model.get("deep_supervision_layers"):
            model.pop("deep_supervision_layers", None)
    # Preserve hashes from checkpoints created before ranked activity-aware
    # selection was added. Non-default ranked selection remains part of the
    # experiment identity.
    mask = resolved["mask"]
    if mask.get("activity_selection_strategy") == "minimum_active_ratio":
        mask.pop("activity_selection_strategy", None)
    if mask.get("activity_topk_fraction") == 0.25:
        mask.pop("activity_topk_fraction", None)
    # Runtime destinations/cadence do not change optimization semantics and may
    # legitimately differ on resume. The seed remains part of the identity.
    resolved["runtime"] = {"seed": resolved["runtime"]["seed"]}
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint_atomic(
    path: str | Path,
    model: WindowJEPA,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
    global_step: int,
    world_size: int,
    steps_per_epoch: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "resolved_config": config.to_dict(),
        "config_hash": config_hash(config),
        "online_encoder": model.online_encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "scale_embedding": model.scale_embedding.state_dict(),
        "target_scale_embedding": model.target_scale_embedding.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "world_size": world_size,
        "steps_per_epoch": steps_per_epoch,
        "rng_state": _rng_state(),
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary_name = handle.name
        torch.save(checkpoint, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_training_checkpoint(
    path: str | Path,
    model: WindowJEPA,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: torch.device,
    world_size: int,
    steps_per_epoch: int,
) -> tuple[int, int]:
    # Training resume accepts only checkpoints produced by this project.
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if checkpoint.get("config_hash") != config_hash(config):
        raise ValueError("checkpoint configuration differs from the current configuration")
    if checkpoint.get("world_size") != world_size:
        raise ValueError("world_size must not change during strict training resume")
    if checkpoint.get("steps_per_epoch") != steps_per_epoch:
        raise ValueError("steps_per_epoch must not change during strict training resume")
    model.online_encoder.load_state_dict(checkpoint["online_encoder"], strict=True)
    model.target_encoder.load_state_dict(checkpoint["target_encoder"], strict=True)
    model.predictor.load_state_dict(checkpoint["predictor"], strict=True)
    model.scale_embedding.load_state_dict(checkpoint["scale_embedding"], strict=True)
    model.target_scale_embedding.load_state_dict(
        checkpoint["target_scale_embedding"], strict=True
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["epoch"]), int(checkpoint["global_step"])


def load_pretrained_model(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[WindowJEPA, ExperimentConfig]:
    """Strict inference/fine-tuning loader that does not require an optimizer."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    config = ExperimentConfig.from_mapping(checkpoint["resolved_config"])
    if checkpoint.get("config_hash") != config_hash(config):
        raise ValueError("checkpoint contains inconsistent configuration metadata")

    # Local import avoids a module cycle: pretrain uses the resume helpers above.
    from event_window_jepa.train.pretrain import build_model

    model = build_model(config)
    model.online_encoder.load_state_dict(checkpoint["online_encoder"], strict=True)
    model.target_encoder.load_state_dict(checkpoint["target_encoder"], strict=True)
    model.predictor.load_state_dict(checkpoint["predictor"], strict=True)
    model.scale_embedding.load_state_dict(checkpoint["scale_embedding"], strict=True)
    model.target_scale_embedding.load_state_dict(
        checkpoint["target_scale_embedding"], strict=True
    )
    model.to(torch.device(device))
    model.eval()
    return model, config
