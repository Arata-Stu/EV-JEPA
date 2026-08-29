from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown keys in {section}: {sorted(unknown)}")


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a YAML boolean, not a string or number")
    return value


def _tuple_of_floats(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(float(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive durations")
    return result


def _pair_of_ints(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    result = (int(value[0]), int(value[1]))
    if min(result) <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return result


def _tuple_of_nonnegative_ints(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of zero-based block indices")
    result = tuple(int(item) for item in value)
    if any(item < 0 for item in result):
        raise ValueError(f"{name} cannot contain negative block indices")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} cannot contain duplicate block indices")
    return result


@dataclass(frozen=True)
class DataConfig:
    manifest: str
    store: str = "hdf5"
    split: str = "train"
    samples_per_epoch: int = 100_000
    batch_size: int = 64
    workers: int = 8
    sequence_sampling: str = "dataset_balanced"
    crop_size: tuple[int, int] | None = None
    horizontal_flip_probability: float = 0.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> DataConfig:
        _reject_unknown(
            values,
            {
                "manifest",
                "store",
                "split",
                "samples_per_epoch",
                "batch_size",
                "workers",
                "sequence_sampling",
                "crop_size",
                "horizontal_flip_probability",
            },
            "data",
        )
        crop = values.get("crop_size")
        return cls(
            manifest=str(values["manifest"]),
            store=str(values.get("store", "hdf5")),
            split=str(values.get("split", "train")),
            samples_per_epoch=int(values.get("samples_per_epoch", 100_000)),
            batch_size=int(values.get("batch_size", 64)),
            workers=int(values.get("workers", 8)),
            sequence_sampling=str(values.get("sequence_sampling", "dataset_balanced")),
            crop_size=None if crop is None else _pair_of_ints(crop, "data.crop_size"),
            horizontal_flip_probability=float(
                values.get("horizontal_flip_probability", 0.5)
            ),
        )

    def __post_init__(self) -> None:
        if self.store not in {"npz", "hdf5"}:
            raise ValueError("data.store must be npz or hdf5")
        if self.samples_per_epoch <= 0 or self.batch_size <= 0 or self.workers < 0:
            raise ValueError("data sizes must be positive and workers must be non-negative")
        if self.sequence_sampling not in {"dataset_balanced", "sequence_balanced"}:
            raise ValueError("data.sequence_sampling is invalid")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")


@dataclass(frozen=True)
class RepresentationConfig:
    kind: str = "voxel_grid"
    temporal_bins: int = 5
    split_polarity: bool = True
    normalization: str = "log1p"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RepresentationConfig:
        _reject_unknown(
            values,
            {"type", "kind", "temporal_bins", "split_polarity", "normalization"},
            "representation",
        )
        return cls(
            kind=str(values.get("type", values.get("kind", "voxel_grid"))),
            temporal_bins=int(values.get("temporal_bins", 5)),
            split_polarity=_boolean(
                values.get("split_polarity", True), "representation.split_polarity"
            ),
            normalization=str(values.get("normalization", "log1p")),
        )

    def __post_init__(self) -> None:
        if self.kind not in {"voxel_grid", "event_image"}:
            raise ValueError("representation.type must be voxel_grid or event_image")
        if self.temporal_bins <= 0:
            raise ValueError("temporal_bins must be positive")
        if not self.split_polarity:
            raise ValueError("the first implementation requires split_polarity=true")
        if self.normalization not in {"none", "log1p", "global_log1p"}:
            raise ValueError("normalization must be none or log1p/global_log1p")

    @property
    def channels(self) -> int:
        return 2 * self.temporal_bins if self.kind == "voxel_grid" else 2


@dataclass(frozen=True)
class WindowsConfig:
    train_ms: tuple[float, ...] = (10.0, 20.0, 40.0, 80.0)
    eval_ms: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0)
    unseen_eval_ms: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0, 120.0)
    target_ms: tuple[float, ...] = (40.0,)
    canonical_ms: float = 40.0
    minimum_ratio: float = 1.5
    direction: str = "any"
    allow_equal: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> WindowsConfig:
        _reject_unknown(
            values,
            {
                "train_ms",
                "eval_ms",
                "unseen_eval_ms",
                "target_ms",
                "canonical_ms",
                "minimum_ratio",
                "direction",
                "allow_equal",
            },
            "windows",
        )
        train_ms = _tuple_of_floats(values.get("train_ms", [10, 20, 40, 80]), "train_ms")
        return cls(
            train_ms=train_ms,
            eval_ms=_tuple_of_floats(
                values.get("eval_ms", [5, 10, 15, 20, 30, 40, 60, 80, 120]),
                "eval_ms",
            ),
            unseen_eval_ms=_tuple_of_floats(
                values.get("unseen_eval_ms", [5, 15, 30, 60, 120]),
                "unseen_eval_ms",
            ),
            target_ms=_tuple_of_floats(values.get("target_ms", train_ms), "target_ms"),
            canonical_ms=float(values.get("canonical_ms", 40.0)),
            minimum_ratio=float(values.get("minimum_ratio", 1.5)),
            direction=str(values.get("direction", "any")),
            allow_equal=_boolean(values.get("allow_equal", False), "windows.allow_equal"),
        )

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for values in (self.train_ms, self.target_ms, self.eval_ms, self.unseen_eval_ms)
            for value in values
        ):
            raise ValueError("all window durations must be positive")
        if self.canonical_ms <= 0 or self.minimum_ratio < 1.0:
            raise ValueError("canonical_ms must be positive and minimum_ratio must be >= 1")
        if self.direction not in {"any", "short_to_long", "long_to_short"}:
            raise ValueError("windows.direction is invalid")
        train = set(self.train_ms) | set(self.target_ms)
        unseen = set(self.unseen_eval_ms)
        if train & unseen:
            raise ValueError("unseen_eval_ms must not overlap train_ms")
        if not unseen.issubset(set(self.eval_ms)):
            raise ValueError("unseen_eval_ms must be a subset of eval_ms")
        if self.canonical_ms not in set(self.target_ms):
            raise ValueError("canonical_ms must be included in target_ms")
        for name, values in (
            ("train_ms", self.train_ms),
            ("target_ms", self.target_ms),
            ("eval_ms", self.eval_ms),
            ("unseen_eval_ms", self.unseen_eval_ms),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"windows.{name} contains duplicate durations")


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "event_vit_v1"
    image_size: tuple[int, int] = (224, 224)
    patch_size: int = 16
    embed_dim: int = 384
    encoder_depth: int = 12
    encoder_heads: int = 6
    predictor_dim: int = 256
    predictor_depth: int = 4
    predictor_heads: int = 8
    scale_dim: int = 128
    scale_fourier_bands: int = 16
    condition_on_scale: bool = True
    deep_supervision_layers: tuple[int, ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ModelConfig:
        _reject_unknown(
            values,
            {
                "architecture",
                "image_size",
                "patch_size",
                "embed_dim",
                "encoder_depth",
                "encoder_heads",
                "predictor_dim",
                "predictor_depth",
                "predictor_heads",
                "scale_dim",
                "scale_fourier_bands",
                "condition_on_scale",
                "deep_supervision_layers",
            },
            "model",
        )
        return cls(
            architecture=str(values.get("architecture", "event_vit_v1")),
            image_size=_pair_of_ints(values.get("image_size", [224, 224]), "model.image_size"),
            patch_size=int(values.get("patch_size", 16)),
            embed_dim=int(values.get("embed_dim", 384)),
            encoder_depth=int(values.get("encoder_depth", 12)),
            encoder_heads=int(values.get("encoder_heads", 6)),
            predictor_dim=int(values.get("predictor_dim", 256)),
            predictor_depth=int(values.get("predictor_depth", 4)),
            predictor_heads=int(values.get("predictor_heads", 8)),
            scale_dim=int(values.get("scale_dim", 128)),
            scale_fourier_bands=int(values.get("scale_fourier_bands", 16)),
            condition_on_scale=_boolean(
                values.get("condition_on_scale", True), "model.condition_on_scale"
            ),
            deep_supervision_layers=_tuple_of_nonnegative_ints(
                values.get("deep_supervision_layers", []),
                "model.deep_supervision_layers",
            ),
        )

    def __post_init__(self) -> None:
        if self.architecture not in {"event_vit_v1", "vjepa2_1"}:
            raise ValueError("model.architecture must be event_vit_v1 or vjepa2_1")
        h, w = self.image_size
        dimensions = (
            self.patch_size,
            self.embed_dim,
            self.encoder_depth,
            self.encoder_heads,
            self.predictor_dim,
            self.predictor_depth,
            self.predictor_heads,
            self.scale_dim,
            self.scale_fourier_bands,
        )
        if min(dimensions) <= 0:
            raise ValueError("all model dimensions and depths must be positive")
        if self.patch_size <= 0 or h % self.patch_size or w % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.embed_dim % self.encoder_heads:
            raise ValueError("embed_dim must be divisible by encoder_heads")
        if self.predictor_dim % self.predictor_heads:
            raise ValueError("predictor_dim must be divisible by predictor_heads")
        if self.deep_supervision_layers and max(self.deep_supervision_layers) >= self.encoder_depth:
            raise ValueError("deep_supervision_layers must be smaller than encoder_depth")
        if self.architecture == "event_vit_v1" and self.deep_supervision_layers:
            raise ValueError("deep_supervision_layers require model.architecture=vjepa2_1")
        if self.architecture == "vjepa2_1" and (
            self.embed_dim // self.encoder_heads
        ) % 4:
            raise ValueError(
                "vjepa2_1 requires the attention head dimension to be divisible by four"
            )


@dataclass(frozen=True)
class RecurrentConfig:
    """Causal sequence-loader and optional recurrent-state settings.

    ``sequence_length`` counts loss-bearing steps.  The dataset prepends
    ``burn_in_steps`` additional windows, so each returned clip contains
    ``burn_in_steps + sequence_length`` windows in total.  Gradients span at
    most ``tbptt_steps`` loss-bearing windows before the state is detached.

    Explicit ``sampling`` modes separate temporal execution from sampling:
    ``random`` resets independently sampled clips, ``stream_reset`` uses the
    same stable causal lanes as ``stream`` but resets every chunk, ``stream``
    carries detached state across chunks, and ``mixed`` combines stream then
    random rows in each per-rank batch. ``clip`` is the legacy random mode.

    ``enabled`` and ``cell`` are kept as backward-compatible aliases for the
    original R0 configuration.  New configurations should select the two
    independent axes explicitly with ``sequence_loader`` and
    ``temporal_model``.  Missing values are resolved in ``__post_init__`` so
    consumers always observe a concrete boolean and model name at runtime.
    """

    enabled: bool | None = None
    cell: str | None = None
    sequence_loader: bool | None = None
    temporal_model: str | None = None
    return_patch_event_activity: bool = False
    kernel_size: int = 3
    sampling: str = "clip"
    stream_ratio: float = 0.5
    window_ms: float = 50.0
    stride_ms: float = 50.0
    sequence_length: int = 8
    burn_in_steps: int = 2
    tbptt_steps: int = 4
    prediction_horizon_steps: int = 0
    recurrent_placement: str = "pre_encoder"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RecurrentConfig:
        _reject_unknown(
            values,
            {
                "enabled",
                "cell",
                "sequence_loader",
                "temporal_model",
                "return_patch_event_activity",
                "kernel_size",
                "sampling",
                "stream_ratio",
                "window_ms",
                "stride_ms",
                "sequence_length",
                "burn_in_steps",
                "tbptt_steps",
                "prediction_horizon_steps",
                "recurrent_placement",
            },
            "recurrent",
        )
        enabled = values.get("enabled")
        cell = values.get("cell")
        sequence_loader = values.get("sequence_loader")
        temporal_model = values.get("temporal_model")
        return cls(
            enabled=(
                None if enabled is None else _boolean(enabled, "recurrent.enabled")
            ),
            cell=None if cell is None else str(cell),
            sequence_loader=(
                None
                if sequence_loader is None
                else _boolean(sequence_loader, "recurrent.sequence_loader")
            ),
            temporal_model=(
                None if temporal_model is None else str(temporal_model)
            ),
            return_patch_event_activity=_boolean(
                values.get("return_patch_event_activity", False),
                "recurrent.return_patch_event_activity",
            ),
            kernel_size=int(values.get("kernel_size", 3)),
            sampling=str(values.get("sampling", "clip")),
            stream_ratio=float(values.get("stream_ratio", 0.5)),
            window_ms=float(values.get("window_ms", 50.0)),
            stride_ms=float(values.get("stride_ms", 50.0)),
            sequence_length=int(values.get("sequence_length", 8)),
            burn_in_steps=int(values.get("burn_in_steps", 2)),
            tbptt_steps=int(values.get("tbptt_steps", 4)),
            prediction_horizon_steps=int(
                values.get("prediction_horizon_steps", 0)
            ),
            recurrent_placement=str(
                values.get("recurrent_placement", "pre_encoder")
            ),
        )

    def __post_init__(self) -> None:
        if self.cell is not None and self.cell not in {"conv_lstm", "conv_gru"}:
            raise ValueError("recurrent.cell must be conv_lstm or conv_gru")
        legacy_enabled = False if self.enabled is None else self.enabled
        legacy_cell = "conv_lstm" if self.cell is None else self.cell
        temporal_model = (
            legacy_cell if legacy_enabled else "feedforward"
        ) if self.temporal_model is None else self.temporal_model
        if temporal_model not in {"feedforward", "conv_lstm", "conv_gru"}:
            raise ValueError(
                "recurrent.temporal_model must be feedforward, conv_lstm, or conv_gru"
            )
        recurrent_enabled = temporal_model != "feedforward"
        if self.enabled is not None and self.enabled != recurrent_enabled:
            raise ValueError(
                "recurrent.enabled contradicts recurrent.temporal_model"
            )
        if (
            recurrent_enabled
            and self.cell is not None
            and self.cell != temporal_model
        ):
            raise ValueError("recurrent.cell contradicts recurrent.temporal_model")
        sequence_loader = (
            recurrent_enabled
            if self.sequence_loader is None
            else self.sequence_loader
        )
        if recurrent_enabled and not sequence_loader:
            raise ValueError(
                "a ConvLSTM/ConvGRU temporal_model requires "
                "recurrent.sequence_loader=true"
            )
        if self.return_patch_event_activity and not sequence_loader:
            raise ValueError(
                "return_patch_event_activity requires "
                "recurrent.sequence_loader=true"
            )

        # Resolve legacy aliases as well as the new explicit switches. Frozen
        # dataclasses still permit normalization during construction.
        object.__setattr__(self, "temporal_model", temporal_model)
        object.__setattr__(self, "sequence_loader", sequence_loader)
        object.__setattr__(self, "enabled", recurrent_enabled)
        object.__setattr__(
            self, "cell", temporal_model if recurrent_enabled else legacy_cell
        )
        if self.sampling not in {
            "clip",
            "random",
            "stream_reset",
            "stream",
            "mixed",
        }:
            raise ValueError(
                "recurrent.sampling must be clip, random, stream_reset, stream, "
                "or mixed"
            )
        if self.sampling == "mixed" and self.stream_ratio != 0.5:
            raise ValueError(
                "recurrent.stream_ratio must be 0.5 for RVT-style mixed 1:1 sampling"
            )
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("recurrent.kernel_size must be a positive odd integer")
        if self.window_ms <= 0 or self.stride_ms <= 0:
            raise ValueError("recurrent window and stride must be positive")
        if self.sequence_length <= 0:
            raise ValueError("recurrent.sequence_length must be positive")
        if self.burn_in_steps < 0:
            raise ValueError("recurrent.burn_in_steps cannot be negative")
        if self.prediction_horizon_steps < 0:
            raise ValueError(
                "recurrent.prediction_horizon_steps cannot be negative"
            )
        if self.recurrent_placement not in {"pre_encoder", "post_encoder"}:
            raise ValueError(
                "recurrent.recurrent_placement must be pre_encoder or post_encoder"
            )
        if not 0 < self.tbptt_steps <= self.sequence_length:
            raise ValueError(
                "recurrent.tbptt_steps must lie in [1, sequence_length]"
            )


@dataclass(frozen=True)
class MaskConfig:
    target_blocks: int = 4
    target_area_range: tuple[float, float] = (0.15, 0.25)
    target_aspect_range: tuple[float, float] = (0.5, 2.0)
    context_keep_ratio: float = 0.60
    activity_aware_probability: float = 0.0
    activity_candidates: int = 32
    minimum_active_target_ratio: float = 0.25
    activity_selection_strategy: str = "minimum_active_ratio"
    activity_topk_fraction: float = 0.25

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> MaskConfig:
        _reject_unknown(
            values,
            {
                "target_blocks",
                "target_area_range",
                "target_aspect_range",
                "context_keep_ratio",
                "activity_aware_probability",
                "activity_candidates",
                "minimum_active_target_ratio",
                "activity_selection_strategy",
                "activity_topk_fraction",
            },
            "mask",
        )
        area = values.get("target_area_range", [0.15, 0.25])
        aspect = values.get("target_aspect_range", [0.5, 2.0])
        return cls(
            target_blocks=int(values.get("target_blocks", 4)),
            target_area_range=(float(area[0]), float(area[1])),
            target_aspect_range=(float(aspect[0]), float(aspect[1])),
            context_keep_ratio=float(values.get("context_keep_ratio", 0.60)),
            activity_aware_probability=float(
                values.get("activity_aware_probability", 0.0)
            ),
            activity_candidates=int(values.get("activity_candidates", 32)),
            minimum_active_target_ratio=float(
                values.get("minimum_active_target_ratio", 0.25)
            ),
            activity_selection_strategy=str(
                values.get("activity_selection_strategy", "minimum_active_ratio")
            ),
            activity_topk_fraction=float(values.get("activity_topk_fraction", 0.25)),
        )

    def __post_init__(self) -> None:
        if self.target_blocks <= 0:
            raise ValueError("target_blocks must be positive")
        if not 0 < self.target_area_range[0] <= self.target_area_range[1] < 1:
            raise ValueError("target_area_range must lie inside (0, 1)")
        if not 0 < self.target_aspect_range[0] <= self.target_aspect_range[1]:
            raise ValueError("target_aspect_range must be positive")
        if not 0 < self.context_keep_ratio < 1:
            raise ValueError("context_keep_ratio must lie inside (0, 1)")
        if not 0 <= self.activity_aware_probability <= 1:
            raise ValueError("activity_aware_probability must lie inside [0, 1]")
        if self.activity_candidates <= 0:
            raise ValueError("activity_candidates must be positive")
        if not 0 <= self.minimum_active_target_ratio <= 1:
            raise ValueError("minimum_active_target_ratio must lie inside [0, 1]")
        if self.activity_selection_strategy not in {
            "minimum_active_ratio",
            "topk_enrichment",
        }:
            raise ValueError(
                "activity_selection_strategy must be minimum_active_ratio or "
                "topk_enrichment"
            )
        if not 0 < self.activity_topk_fraction <= 1:
            raise ValueError("activity_topk_fraction must lie inside (0, 1]")
        if self.context_keep_ratio + self.target_area_range[1] > 1.0:
            raise ValueError("context and maximum target area cannot be disjoint")


@dataclass(frozen=True)
class OptimizationConfig:
    objective: str = "window_jepa"
    epochs: int = 100
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 1e-6
    warmup_epochs: int = 10
    weight_decay: float = 0.05
    target_ema_start: float = 0.996
    target_ema_end: float = 1.0
    precision: str = "bf16"
    gradient_clip: float = 1.0
    variance_weight: float = 0.0
    covariance_weight: float = 0.0
    canonical_query_weight: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> OptimizationConfig:
        _reject_unknown(
            values,
            {
                "objective",
                "epochs",
                "learning_rate",
                "minimum_learning_rate",
                "warmup_epochs",
                "weight_decay",
                "target_ema_start",
                "target_ema_end",
                "precision",
                "gradient_clip",
                "variance_weight",
                "covariance_weight",
                "canonical_query_weight",
            },
            "optimization",
        )
        return cls(
            objective=str(values.get("objective", "window_jepa")),
            epochs=int(values.get("epochs", 100)),
            learning_rate=float(values.get("learning_rate", 3e-4)),
            minimum_learning_rate=float(values.get("minimum_learning_rate", 1e-6)),
            warmup_epochs=int(values.get("warmup_epochs", 10)),
            weight_decay=float(values.get("weight_decay", 0.05)),
            target_ema_start=float(values.get("target_ema_start", 0.996)),
            target_ema_end=float(values.get("target_ema_end", 1.0)),
            precision=str(values.get("precision", "bf16")),
            gradient_clip=float(values.get("gradient_clip", 1.0)),
            variance_weight=float(values.get("variance_weight", 0.0)),
            covariance_weight=float(values.get("covariance_weight", 0.0)),
            canonical_query_weight=float(values.get("canonical_query_weight", 0.0)),
        )

    def __post_init__(self) -> None:
        if self.objective not in {
            "window_jepa",
            "dense_window_jepa",
            "sequence_window_jepa",
            "sequence_dense_window_jepa",
            "recurrent_window_jepa",
            "recurrent_dense_window_jepa",
            "recurrent_future_jepa",
            "feature_consistency",
        }:
            raise ValueError(
                "objective must be a paired/sequence/recurrent Window-JEPA objective "
                "or feature_consistency"
            )
        if self.epochs <= 0 or self.warmup_epochs < 0:
            raise ValueError("epochs must be positive and warmup_epochs non-negative")
        if self.learning_rate <= 0 or self.minimum_learning_rate < 0:
            raise ValueError("learning rates must be non-negative")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        if (
            self.weight_decay < 0
            or self.gradient_clip <= 0
            or self.variance_weight < 0
            or self.covariance_weight < 0
            or self.canonical_query_weight < 0
        ):
            raise ValueError("optimization regularizers are invalid")
        if not 0 <= self.target_ema_start <= self.target_ema_end <= 1:
            raise ValueError("EMA momentum must increase within [0, 1]")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")


@dataclass(frozen=True)
class FuturePredictionConfig:
    """Collapse control and event-support settings for causal future JEPA."""

    active_min_events: int = 1
    activity_floor: float = 0.01
    frame_sigreg_weight: float = 0.0
    temporal_sigreg_weight: float = 0.0
    allow_unregularized: bool = False
    projector_hidden_dim: int = 512
    projector_output_dim: int = 256
    sigreg_num_slices: int = 1024
    sigreg_t_max: float = 3.0
    sigreg_num_points: int = 17
    projection_seed: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> FuturePredictionConfig:
        _reject_unknown(
            values,
            {
                "active_min_events",
                "activity_floor",
                "frame_sigreg_weight",
                "temporal_sigreg_weight",
                "allow_unregularized",
                "projector_hidden_dim",
                "projector_output_dim",
                "sigreg_num_slices",
                "sigreg_t_max",
                "sigreg_num_points",
                "projection_seed",
            },
            "future_prediction",
        )
        return cls(
            active_min_events=int(values.get("active_min_events", 1)),
            activity_floor=float(values.get("activity_floor", 0.01)),
            frame_sigreg_weight=float(values.get("frame_sigreg_weight", 0.0)),
            temporal_sigreg_weight=float(
                values.get("temporal_sigreg_weight", 0.0)
            ),
            allow_unregularized=_boolean(
                values.get("allow_unregularized", False),
                "future_prediction.allow_unregularized",
            ),
            projector_hidden_dim=int(values.get("projector_hidden_dim", 512)),
            projector_output_dim=int(values.get("projector_output_dim", 256)),
            sigreg_num_slices=int(values.get("sigreg_num_slices", 1024)),
            sigreg_t_max=float(values.get("sigreg_t_max", 3.0)),
            sigreg_num_points=int(values.get("sigreg_num_points", 17)),
            projection_seed=int(values.get("projection_seed", 0)),
        )

    def __post_init__(self) -> None:
        if self.active_min_events <= 0:
            raise ValueError("future_prediction.active_min_events must be positive")
        if not 0 < self.activity_floor <= 1:
            raise ValueError(
                "future_prediction.activity_floor must lie inside (0, 1]"
            )
        if self.frame_sigreg_weight < 0 or self.temporal_sigreg_weight < 0:
            raise ValueError("future_prediction SIGReg weights cannot be negative")
        if min(self.projector_hidden_dim, self.projector_output_dim) <= 0:
            raise ValueError("future_prediction projector dimensions must be positive")
        if self.sigreg_num_slices <= 0:
            raise ValueError("future_prediction.sigreg_num_slices must be positive")
        if self.sigreg_t_max <= 0:
            raise ValueError("future_prediction.sigreg_t_max must be positive")
        if self.sigreg_num_points < 2:
            raise ValueError(
                "future_prediction.sigreg_num_points must be at least two"
            )
        if self.projection_seed < 0:
            raise ValueError("future_prediction.projection_seed cannot be negative")


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 0
    output_dir: str = "outputs/window_jepa"
    log_every_steps: int = 20
    checkpoint_every_epochs: int = 1
    resume: str | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RuntimeConfig:
        _reject_unknown(
            values,
            {
                "seed",
                "output_dir",
                "log_every_steps",
                "checkpoint_every_epochs",
                "resume",
            },
            "runtime",
        )
        return cls(
            seed=int(values.get("seed", 0)),
            output_dir=str(values.get("output_dir", "outputs/window_jepa")),
            log_every_steps=int(values.get("log_every_steps", 20)),
            checkpoint_every_epochs=int(values.get("checkpoint_every_epochs", 1)),
            resume=values.get("resume"),
        )

    def __post_init__(self) -> None:
        if self.log_every_steps <= 0 or self.checkpoint_every_epochs <= 0:
            raise ValueError("logging and checkpoint intervals must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    future_prediction: FuturePredictionConfig = field(
        default_factory=FuturePredictionConfig
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        if self.data.crop_size is None:
            raise ValueError("data.crop_size is required by the fixed-size training pipeline")
        if self.data.crop_size != self.model.image_size:
            raise ValueError("data.crop_size must equal model.image_size")
        if self.data.batch_size < 2:
            raise ValueError("per-rank batch_size must be at least 2 for collapse diagnostics")
        if self.optimization.objective in {
            "window_jepa",
            "dense_window_jepa",
            "sequence_window_jepa",
            "sequence_dense_window_jepa",
            "recurrent_window_jepa",
            "recurrent_dense_window_jepa",
            "recurrent_future_jepa",
        }:
            if self.optimization.covariance_weight:
                raise ValueError("covariance_weight is only used by feature_consistency")
            if (
                self.optimization.variance_weight
                and not self.optimization.canonical_query_weight
            ):
                raise ValueError(
                    "Window-JEPA variance loss requires canonical_query_weight > 0 "
                    "for fixed-support batch statistics"
                )
        else:
            if self.optimization.canonical_query_weight:
                raise ValueError("canonical_query_weight is only used by window_jepa")
            if not (
                self.optimization.variance_weight or self.optimization.covariance_weight
            ):
                raise ValueError(
                    "feature_consistency requires variance or covariance anti-collapse loss"
                )
        if (
            self.optimization.objective in {
                "dense_window_jepa",
                "sequence_dense_window_jepa",
            }
            and self.model.architecture != "vjepa2_1"
        ):
            raise ValueError(
                "dense Window-JEPA objectives require model.architecture=vjepa2_1"
            )
        if (
            self.model.architecture == "vjepa2_1"
            and self.model.deep_supervision_layers
            and self.model.encoder_depth - 1
            not in self.model.deep_supervision_layers
        ):
            raise ValueError(
                "explicit V-JEPA 2.1 supervision must include the final "
                "encoder block so DDP has no unused trailing blocks"
            )
        sequence_objectives = {
            "sequence_window_jepa",
            "sequence_dense_window_jepa",
        }
        recurrent_objectives = {
            "recurrent_window_jepa",
            "recurrent_dense_window_jepa",
            "recurrent_future_jepa",
        }
        paired_objectives = {
            "window_jepa",
            "dense_window_jepa",
            "feature_consistency",
        }
        objective = self.optimization.objective
        is_future_objective = objective == "recurrent_future_jepa"
        is_sequence_objective = objective in sequence_objectives
        is_recurrent_objective = self.optimization.objective in recurrent_objectives
        if objective in paired_objectives and self.recurrent.sequence_loader:
            raise ValueError(
                "paired objectives require recurrent.sequence_loader=false"
            )
        if is_sequence_objective and (
            not self.recurrent.sequence_loader or self.recurrent.enabled
        ):
            raise ValueError(
                "sequence objectives require recurrent.sequence_loader=true and "
                "recurrent.temporal_model=feedforward"
            )
        if is_recurrent_objective and (
            not self.recurrent.sequence_loader or not self.recurrent.enabled
        ):
            raise ValueError(
                "recurrent objectives require recurrent.sequence_loader=true and "
                "a ConvLSTM/ConvGRU temporal_model"
            )
        if self.recurrent.enabled:
            if self.model.architecture != "vjepa2_1":
                raise ValueError("R0 recurrent pretraining requires model.architecture=vjepa2_1")
        if is_future_objective:
            if self.recurrent.prediction_horizon_steps < 1:
                raise ValueError(
                    "recurrent_future_jepa requires prediction_horizon_steps >= 1"
                )
            if self.recurrent.recurrent_placement != "post_encoder":
                raise ValueError(
                    "recurrent_future_jepa requires recurrent_placement=post_encoder"
                )
            if not self.recurrent.return_patch_event_activity:
                raise ValueError(
                    "recurrent_future_jepa requires return_patch_event_activity=true"
                )
            if self.recurrent.burn_in_steps < 1:
                raise ValueError(
                    "recurrent_future_jepa requires at least one burn-in step"
                )
            if self.model.deep_supervision_layers != (
                self.model.encoder_depth - 1,
            ):
                raise ValueError(
                    "recurrent_future_jepa supervises only the final frame latent; "
                    "model.deep_supervision_layers must contain only the final block"
                )
            if not (
                self.future_prediction.frame_sigreg_weight
                or self.future_prediction.temporal_sigreg_weight
                or self.future_prediction.allow_unregularized
            ):
                raise ValueError(
                    "recurrent_future_jepa requires SIGReg unless "
                    "future_prediction.allow_unregularized=true is set for an ablation"
                )
        else:
            if (
                self.future_prediction.frame_sigreg_weight
                or self.future_prediction.temporal_sigreg_weight
            ):
                raise ValueError(
                    "future SIGReg weights are only used by recurrent_future_jepa"
                )
            if self.recurrent.prediction_horizon_steps:
                raise ValueError(
                    "prediction_horizon_steps is only used by recurrent_future_jepa"
                )
            if (
                self.recurrent.enabled
                and self.recurrent.recurrent_placement != "pre_encoder"
            ):
                raise ValueError(
                    "legacy recurrent objectives require recurrent_placement=pre_encoder"
                )
        if self.recurrent.sequence_loader:
            expected_windows = (self.recurrent.window_ms,)
            if (
                self.windows.train_ms != expected_windows
                or self.windows.target_ms != expected_windows
                or not self.windows.allow_equal
            ):
                raise ValueError(
                    "the sequence loader requires train_ms and target_ms to contain only "
                    "recurrent.window_ms, with windows.allow_equal=true"
                )
            if self.windows.canonical_ms != self.recurrent.window_ms:
                raise ValueError(
                    "the sequence loader requires windows.canonical_ms == "
                    "recurrent.window_ms"
                )
            if self.recurrent.stride_ms != self.recurrent.window_ms:
                raise ValueError(
                    "the sequence loader requires stride_ms == window_ms so "
                    "consecutive event bins "
                    "neither overlap nor leave temporal gaps"
                )
            if self.recurrent.enabled and self.optimization.canonical_query_weight:
                raise ValueError(
                    "R0 requires canonical_query_weight=0 because a second full-context "
                    "online pass would leak the current masked input into recurrent state"
                )
            if self.recurrent.sampling == "mixed":
                if self.data.batch_size % 2:
                    raise ValueError(
                        "RVT-style mixed 1:1 sampling requires an even per-rank "
                        "batch size"
                    )
                stream_batch = round(
                    self.data.batch_size * self.recurrent.stream_ratio
                )
                if not 0 < stream_batch < self.data.batch_size:
                    raise ValueError(
                        "mixed sequence sampling requires at least one stream and one "
                        "random sample per rank"
                    )
            if (
                self.recurrent.enabled
                and self.recurrent.sampling
                in {"random", "stream_reset", "stream", "mixed"}
                and self.recurrent.tbptt_steps
                != self.recurrent.sequence_length
            ):
                raise ValueError(
                    "explicit random/stream_reset/stream/mixed recurrent sampling "
                    "requires "
                    "tbptt_steps == sequence_length: random clips use full BPTT, "
                    "while stream state is detached and carried between batches"
                )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ExperimentConfig:
        _reject_unknown(
            values,
            {
                "data",
                "representation",
                "windows",
                "model",
                "recurrent",
                "mask",
                "optimization",
                "future_prediction",
                "runtime",
            },
            "configuration root",
        )
        return cls(
            data=DataConfig.from_mapping(values["data"]),
            representation=RepresentationConfig.from_mapping(values.get("representation", {})),
            windows=WindowsConfig.from_mapping(values.get("windows", {})),
            model=ModelConfig.from_mapping(values.get("model", {})),
            recurrent=RecurrentConfig.from_mapping(values.get("recurrent", {})),
            mask=MaskConfig.from_mapping(values.get("mask", {})),
            optimization=OptimizationConfig.from_mapping(values.get("optimization", {})),
            future_prediction=FuturePredictionConfig.from_mapping(
                values.get("future_prediction", {})
            ),
            runtime=RuntimeConfig.from_mapping(values.get("runtime", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, Mapping):
            raise ValueError("configuration root must be a mapping")
        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
