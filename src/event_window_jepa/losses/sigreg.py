from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch import nn


def _fp32_context(features: torch.Tensor) -> Any:
    """Disable the training autocast region around SIGReg numerics."""

    if features.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=features.device.type, enabled=False)
    return nullcontext()


def _autograd_all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    """Sum a tensor over ranks while preserving its backward graph.

    This helper intentionally does not inspect event support. Every caller must
    invoke it in the same order on every rank, including ranks whose local valid
    mask is entirely false.
    """

    if (
        distributed.is_available()
        and distributed.is_initialized()
        and distributed.get_world_size() > 1
    ):
        from torch.distributed.nn.functional import all_reduce

        return all_reduce(value, op=distributed.ReduceOp.SUM)
    return value


@dataclass(frozen=True)
class SIGRegOutput:
    """Sliced Epps--Pulley loss and global-batch diagnostics."""

    loss: torch.Tensor
    real_error: torch.Tensor
    imaginary_error: torch.Tensor
    effective_samples: torch.Tensor


class SIGRegProjector(nn.Module):
    """Small deterministic MLP whose forward pass always runs in FP32.

    ``seed`` is independent of the process-global RNG. Thus projectors created
    with the same configuration start identically even when training ranks use
    different data RNG seeds. DDP still owns the usual parameter synchronization
    and gradient reduction.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        *,
        hidden_dim: int | None = None,
        depth: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("SIGReg projector dimensions must be positive")
        if depth <= 0:
            raise ValueError("SIGReg projector depth must be positive")
        if seed < 0:
            raise ValueError("SIGReg projector seed cannot be negative")
        hidden = max(input_dim, output_dim) if hidden_dim is None else hidden_dim
        if hidden <= 0:
            raise ValueError("SIGReg projector hidden_dim must be positive")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.depth = int(depth)
        self.seed = int(seed)

        # Seed only the CPU default generator. ``torch.manual_seed`` would also
        # overwrite every rank-local CUDA generator, while forking all visible
        # GPUs could initialize foreign DDP devices in each process.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(self.seed)
            layers: list[nn.Module] = []
            if self.depth == 1:
                layers.append(nn.Linear(self.input_dim, self.output_dim))
            else:
                layers.extend((nn.Linear(self.input_dim, hidden), nn.GELU()))
                for _ in range(self.depth - 2):
                    layers.extend((nn.Linear(hidden, hidden), nn.GELU()))
                layers.append(nn.Linear(hidden, self.output_dim))
            self.network = nn.Sequential(*layers)
            for module in self.network.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"SIGReg projector expects [B,{self.input_dim}], got {tuple(features.shape)}"
            )
        if features.shape[0] <= 0:
            raise ValueError("SIGReg projector requires a non-empty batch")
        if not features.is_floating_point():
            raise TypeError("SIGReg projector features must be floating point")
        first_parameter = next(self.parameters())
        if first_parameter.device != features.device:
            raise ValueError("SIGReg projector and features must share a device")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise TypeError("SIGReg projector parameters must remain FP32")
        with _fp32_context(features):
            return self.network(features.float()).float()


class SlicedEppsPulleySIGReg(nn.Module):
    """Match random one-dimensional projections to a standard normal ECF.

    Unit directions and positive frequency knots are fixed buffers constructed
    from an explicit CPU generator. They are therefore identical on all ranks
    without consuming rank-local RNG state. The empirical characteristic
    function is reduced over the global batch with an autograd-aware sum.

    The feature input is deliberately restricted to ``[B,D]``. Temporal callers
    must invoke the module once per timestep instead of flattening ``B x T``.
    ``valid_mask`` only removes zero-support rows; it never changes the number or
    order of distributed collectives.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        num_slices: int = 128,
        num_frequencies: int = 17,
        maximum_frequency: float = 3.0,
        minimum_samples: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or num_slices <= 0 or num_frequencies <= 0:
            raise ValueError("SIGReg feature, slice, and frequency counts must be positive")
        if maximum_frequency <= 0:
            raise ValueError("SIGReg maximum_frequency must be positive")
        if minimum_samples < 2:
            raise ValueError("SIGReg minimum_samples must be at least two")
        if seed < 0:
            raise ValueError("SIGReg seed cannot be negative")

        self.feature_dim = int(feature_dim)
        self.num_slices = int(num_slices)
        self.num_frequencies = int(num_frequencies)
        self.maximum_frequency = float(maximum_frequency)
        self.minimum_samples = int(minimum_samples)
        self.seed = int(seed)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        directions = torch.randn(
            self.num_slices,
            self.feature_dim,
            generator=generator,
            dtype=torch.float32,
        )
        directions = functional.normalize(directions, dim=1)
        frequencies = torch.linspace(
            self.maximum_frequency / self.num_frequencies,
            self.maximum_frequency,
            self.num_frequencies,
            dtype=torch.float32,
        )
        # Gaussian integration weights emphasize the informative low-frequency
        # characteristic function while still testing non-Gaussian tails.
        frequency_weights = torch.exp(-0.5 * frequencies.square())
        frequency_weights = frequency_weights / frequency_weights.sum()
        target_characteristic = torch.exp(-0.5 * frequencies.square())
        self.register_buffer("directions", directions)
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("frequency_weights", frequency_weights)
        self.register_buffer("target_characteristic", target_characteristic)

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> SIGRegOutput:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"SIGReg expects [B,{self.feature_dim}], got {tuple(features.shape)}"
            )
        if features.shape[0] <= 0:
            raise ValueError("SIGReg requires a non-empty local batch")
        if not features.is_floating_point():
            raise TypeError("SIGReg features must be floating point")
        if self.directions.device != features.device:
            raise ValueError("SIGReg buffers and features must share a device")
        if valid_mask is None:
            valid_mask = torch.ones(
                features.shape[0],
                dtype=torch.bool,
                device=features.device,
            )
        elif valid_mask.dtype != torch.bool or valid_mask.shape != (
            features.shape[0],
        ):
            raise ValueError("valid_mask must be boolean with shape [B]")
        elif valid_mask.device != features.device:
            raise ValueError("valid_mask and features must share a device")

        with _fp32_context(features):
            values = torch.where(
                valid_mask.unsqueeze(1),
                features.float(),
                torch.zeros_like(features, dtype=torch.float32),
            )
            directions = self.directions.float()
            frequencies = self.frequencies.float()
            projected = values @ directions.transpose(0, 1)
            angles = projected.unsqueeze(-1) * frequencies.reshape(1, 1, -1)
            support = valid_mask.to(torch.float32).reshape(-1, 1, 1)

            local_count = support[:, 0, 0].sum()
            local_real = (angles.cos() * support).sum(dim=0)
            local_imaginary = (angles.sin() * support).sum(dim=0)

            # These collectives are unconditional with respect to local support.
            global_count = _autograd_all_reduce_sum(local_count)
            global_real = _autograd_all_reduce_sum(local_real)
            global_imaginary = _autograd_all_reduce_sum(local_imaginary)

            denominator = global_count.clamp_min(1.0)
            empirical_real = global_real / denominator
            empirical_imaginary = global_imaginary / denominator
            target = self.target_characteristic.float().reshape(1, -1)
            weights = self.frequency_weights.float().reshape(1, -1)
            real_error = ((empirical_real - target).square() * weights).sum(dim=1)
            imaginary_error = (empirical_imaginary.square() * weights).sum(dim=1)
            real_error = real_error.mean()
            imaginary_error = imaginary_error.mean()

            # A global, not local, gate makes every rank take the same path. The
            # multiplication retains a zero-gradient graph for empty projectors.
            enough_samples = (global_count >= self.minimum_samples).to(torch.float32)
            real_error = real_error * enough_samples
            imaginary_error = imaginary_error * enough_samples
            loss = real_error + imaginary_error

        return SIGRegOutput(
            loss=loss.float(),
            real_error=real_error.float(),
            imaginary_error=imaginary_error.float(),
            effective_samples=global_count.detach().float(),
        )


class ProjectedSIGReg(nn.Module):
    """Convenience module combining an FP32 projector and sliced SIGReg."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        *,
        hidden_dim: int | None = None,
        projector_depth: int = 2,
        num_slices: int = 128,
        num_frequencies: int = 17,
        maximum_frequency: float = 3.0,
        minimum_samples: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.projector = SIGRegProjector(
            input_dim,
            projection_dim,
            hidden_dim=hidden_dim,
            depth=projector_depth,
            seed=seed,
        )
        self.sigreg = SlicedEppsPulleySIGReg(
            projection_dim,
            num_slices=num_slices,
            num_frequencies=num_frequencies,
            maximum_frequency=maximum_frequency,
            minimum_samples=minimum_samples,
            seed=seed + 1,
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> SIGRegOutput:
        safe_features = features
        if valid_mask is not None:
            if valid_mask.dtype is not torch.bool or valid_mask.shape != (
                features.shape[0],
            ):
                raise ValueError("valid_mask must be boolean with shape [B]")
            if valid_mask.device != features.device:
                raise ValueError("valid_mask and features must share a device")
            safe_features = torch.where(
                valid_mask.unsqueeze(1),
                features,
                torch.zeros_like(features),
            )
        return self.sigreg(
            self.projector(safe_features), valid_mask=valid_mask
        )


__all__ = [
    "ProjectedSIGReg",
    "SIGRegOutput",
    "SIGRegProjector",
    "SlicedEppsPulleySIGReg",
]
