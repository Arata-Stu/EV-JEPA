from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch import nn

from event_window_jepa.losses.latent_prediction import (
    balanced_event_support_latent_prediction_loss,
    latent_prediction_loss,
)
from event_window_jepa.losses.sigreg import ProjectedSIGReg
from event_window_jepa.losses.variance_regularization import (
    covariance_regularization,
    feature_standard_deviation,
    masked_position_standard_deviation,
    masked_position_variance_regularization,
    variance_regularization,
)
from event_window_jepa.models.ema_encoder import make_ema_copy, update_ema
from event_window_jepa.models.event_vit import EventVisionTransformer
from event_window_jepa.models.recurrent_vjepa21_event_vit import (
    RecurrentState,
    RecurrentVJEPA21EventVisionTransformer,
    detach_recurrent_state,
)
from event_window_jepa.models.scale_embedding import LogFourierScaleEmbedding
from event_window_jepa.models.vjepa21_event_vit import VJEPA21EventVisionTransformer
from event_window_jepa.models.window_predictor import WindowPredictor


@dataclass
class WindowJEPAOutput:
    loss: torch.Tensor
    masked_loss: torch.Tensor
    canonical_loss: torch.Tensor
    dense_loss: torch.Tensor
    visible_loss: torch.Tensor
    deep_supervision_loss: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    prediction_std: torch.Tensor
    target_std: torch.Tensor
    online_state: RecurrentState | None = None
    prediction_sequence: torch.Tensor | None = None
    target_sequence: torch.Tensor | None = None
    future_prediction_loss: torch.Tensor | None = None
    active_prediction_loss: torch.Tensor | None = None
    inactive_prediction_loss: torch.Tensor | None = None
    frame_sigreg_loss: torch.Tensor | None = None
    support_sigreg_loss: torch.Tensor | None = None
    temporal_sigreg_loss: torch.Tensor | None = None
    sigreg_loss: torch.Tensor | None = None
    active_patch_fraction: torch.Tensor | None = None
    context_active_patch_fraction: torch.Tensor | None = None
    frame_sigreg_samples: torch.Tensor | None = None
    support_sigreg_samples: torch.Tensor | None = None
    temporal_sigreg_samples: torch.Tensor | None = None
    frame_sigreg_real_error: torch.Tensor | None = None
    frame_sigreg_imaginary_error: torch.Tensor | None = None
    support_sigreg_real_error: torch.Tensor | None = None
    support_sigreg_imaginary_error: torch.Tensor | None = None
    temporal_sigreg_real_error: torch.Tensor | None = None
    temporal_sigreg_imaginary_error: torch.Tensor | None = None
    active_prediction_sum: torch.Tensor | None = None
    active_prediction_count: torch.Tensor | None = None
    inactive_prediction_sum: torch.Tensor | None = None
    inactive_prediction_count: torch.Tensor | None = None


class WindowJEPA(nn.Module):
    def __init__(
        self,
        encoder: (
            EventVisionTransformer
            | VJEPA21EventVisionTransformer
            | RecurrentVJEPA21EventVisionTransformer
        ),
        predictor: WindowPredictor,
        scale_embedding: LogFourierScaleEmbedding,
        condition_on_scale: bool = True,
        loss_kind: str = "smooth_l1",
        variance_weight: float = 0.0,
        covariance_weight: float = 0.0,
        canonical_query_weight: float = 0.0,
        future_active_min_events: int = 1,
        future_activity_floor: float = 0.01,
        frame_sigreg_weight: float = 0.0,
        temporal_sigreg_weight: float = 0.0,
        sigreg_projector_hidden_dim: int = 512,
        sigreg_projector_output_dim: int = 256,
        sigreg_num_slices: int = 1024,
        sigreg_t_max: float = 3.0,
        sigreg_num_points: int = 17,
        sigreg_projection_seed: int = 0,
    ) -> None:
        super().__init__()
        if predictor.num_patches != encoder.num_patches:
            raise ValueError("encoder and predictor patch grids do not match")
        if predictor.encoder_dim != encoder.embed_dim:
            raise ValueError("predictor output and encoder feature dimensions do not match")
        if not (
            encoder.scale_dim == predictor.scale_dim == scale_embedding.output_dim
        ):
            raise ValueError("scale embedding dimensions do not match model consumers")
        if variance_weight < 0 or covariance_weight < 0:
            raise ValueError("regularization weights cannot be negative")
        if canonical_query_weight < 0:
            raise ValueError("canonical_query_weight cannot be negative")
        if future_active_min_events <= 0:
            raise ValueError("future_active_min_events must be positive")
        if not 0 < future_activity_floor <= 1:
            raise ValueError("future_activity_floor must lie inside (0, 1]")
        if frame_sigreg_weight < 0 or temporal_sigreg_weight < 0:
            raise ValueError("future SIGReg weights cannot be negative")
        self.online_encoder = encoder
        self.target_encoder = make_ema_copy(encoder)
        self.predictor = predictor
        self.scale_embedding = scale_embedding
        self.target_scale_embedding = make_ema_copy(scale_embedding)
        self.condition_on_scale = condition_on_scale
        self.loss_kind = loss_kind
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.canonical_query_weight = canonical_query_weight
        self.future_active_min_events = int(future_active_min_events)
        self.future_activity_floor = float(future_activity_floor)
        self.frame_sigreg_weight = float(frame_sigreg_weight)
        self.temporal_sigreg_weight = float(temporal_sigreg_weight)
        self.future_regularizers = nn.ModuleDict()
        if self.frame_sigreg_weight:
            self.future_regularizers["frame"] = ProjectedSIGReg(
                encoder.embed_dim,
                sigreg_projector_output_dim,
                hidden_dim=sigreg_projector_hidden_dim,
                num_slices=sigreg_num_slices,
                num_frequencies=sigreg_num_points,
                maximum_frequency=sigreg_t_max,
                seed=sigreg_projection_seed,
            )
            self.future_regularizers["support"] = ProjectedSIGReg(
                encoder.embed_dim,
                sigreg_projector_output_dim,
                hidden_dim=sigreg_projector_hidden_dim,
                num_slices=sigreg_num_slices,
                num_frequencies=sigreg_num_points,
                maximum_frequency=sigreg_t_max,
                seed=sigreg_projection_seed + 5_000,
            )
        if self.temporal_sigreg_weight:
            self.future_regularizers["temporal"] = ProjectedSIGReg(
                encoder.embed_dim,
                sigreg_projector_output_dim,
                hidden_dim=sigreg_projector_hidden_dim,
                num_slices=sigreg_num_slices,
                num_frequencies=sigreg_num_points,
                maximum_frequency=sigreg_t_max,
                seed=sigreg_projection_seed + 10_000,
            )

    @property
    def num_patches(self) -> int:
        return self.online_encoder.num_patches

    def train(self, mode: bool = True) -> WindowJEPA:
        super().train(mode)
        self.target_encoder.eval()
        self.target_scale_embedding.eval()
        return self

    def auxiliary_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return trainable future-objective state omitted by legacy checkpoints."""

        return {"future_regularizers": self.future_regularizers.state_dict()}

    def load_auxiliary_state_dict(
        self,
        state: dict[str, object],
        *,
        strict: bool = True,
    ) -> None:
        expected = bool(self.future_regularizers.state_dict())
        saved = state.get("future_regularizers") if isinstance(state, dict) else None
        if saved is None:
            if strict and expected:
                raise ValueError(
                    "checkpoint is missing future SIGReg projector state"
                )
            return
        if not isinstance(saved, dict):
            raise TypeError("future_regularizers checkpoint state must be a mapping")
        self.future_regularizers.load_state_dict(saved, strict=strict)

    def _scale_features(
        self, context_ms: torch.Tensor, target_ms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.scale_embedding(context_ms)
        target = self.scale_embedding(target_ms)
        ratio = self.scale_embedding.ratio(target_ms, context_ms)
        if not self.condition_on_scale:
            context = torch.zeros_like(context)
            target = torch.zeros_like(target)
            ratio = torch.zeros_like(ratio)
        return context, target, ratio

    def forward(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        dt_context_ms: torch.Tensor,
        dt_target_ms: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        objective: str = "window_jepa",
        online_state: RecurrentState | None = None,
        sequence_loss_mask: torch.Tensor | None = None,
        context_event_activity: torch.Tensor | None = None,
        target_event_activity: torch.Tensor | None = None,
    ) -> WindowJEPAOutput:
        if objective in {"sequence_window_jepa", "sequence_dense_window_jepa"}:
            if sequence_loss_mask is None:
                raise ValueError("sequence objectives require sequence_loss_mask [B,T]")
            return self.feedforward_sequence(
                x_context=x_context,
                x_target=x_target,
                context_duration_ms=dt_context_ms,
                target_duration_ms=dt_target_ms,
                context_mask=context_mask,
                target_mask=target_mask,
                loss_mask=sequence_loss_mask,
                dense=objective == "sequence_dense_window_jepa",
            )
        if objective in {"recurrent_window_jepa", "recurrent_dense_window_jepa"}:
            return self.recurrent_sequence(
                x_context=x_context,
                x_target=x_target,
                context_duration_ms=dt_context_ms,
                target_duration_ms=dt_target_ms,
                context_mask=context_mask,
                target_mask=target_mask,
                online_state=online_state,
                dense=objective == "recurrent_dense_window_jepa",
            )
        if objective == "recurrent_future_jepa":
            if context_event_activity is None or target_event_activity is None:
                raise ValueError(
                    "recurrent_future_jepa requires context and target event activity"
                )
            return self.recurrent_future_sequence(
                x_context=x_context,
                x_target=x_target,
                context_duration_ms=dt_context_ms,
                target_duration_ms=dt_target_ms,
                context_event_activity=context_event_activity,
                target_event_activity=target_event_activity,
                online_state=online_state,
            )
        if objective == "feature_consistency":
            return self.feature_consistency(
                x_context=x_context,
                x_target=x_target,
                dt_context_ms=dt_context_ms,
                dt_target_ms=dt_target_ms,
            )
        if objective == "dense_window_jepa":
            return self.dense_window_jepa(
                x_context=x_context,
                x_target=x_target,
                dt_context_ms=dt_context_ms,
                dt_target_ms=dt_target_ms,
                context_mask=context_mask,
                target_mask=target_mask,
            )
        if objective != "window_jepa":
            raise ValueError("unsupported pretraining objective")
        if x_context.shape != x_target.shape or x_context.ndim != 4:
            raise ValueError("context and target inputs must share shape [B, C, H, W]")
        batch_size = x_context.shape[0]
        context_ms = dt_context_ms.reshape(batch_size)
        target_ms = dt_target_ms.reshape(batch_size)
        source_scale, target_scale, ratio_scale = self._scale_features(context_ms, target_ms)

        context_tokens = self.online_encoder(x_context, source_scale, context_mask)
        with torch.no_grad():
            ema_target_scale = self.target_scale_embedding(target_ms)
            if not self.condition_on_scale:
                ema_target_scale = torch.zeros_like(ema_target_scale)
            target_tokens = self.target_encoder(x_target, ema_target_scale, None)
        prediction = self.predictor(
            context_tokens=context_tokens,
            context_keep_mask=context_mask,
            target_mask=target_mask,
            source_scale=source_scale,
            target_scale=target_scale,
            ratio_scale=ratio_scale,
        )
        masked_loss = latent_prediction_loss(
            prediction, target_tokens, target_mask=target_mask, kind=self.loss_kind
        )
        canonical_loss = masked_loss.new_zeros(())
        canonical_prediction = prediction
        if self.canonical_query_weight:
            full_mask = torch.ones_like(context_mask)
            full_context_tokens = self.online_encoder(x_context, source_scale, None)
            canonical_prediction = self.predictor(
                context_tokens=full_context_tokens,
                context_keep_mask=full_mask,
                target_mask=full_mask,
                source_scale=source_scale,
                target_scale=target_scale,
                ratio_scale=ratio_scale,
            )
            canonical_loss = latent_prediction_loss(
                canonical_prediction, target_tokens, target_mask=None, kind=self.loss_kind
            )
        loss = masked_loss + self.canonical_query_weight * canonical_loss
        if self.variance_weight:
            full_mask = torch.ones_like(target_mask)
            loss = loss + self.variance_weight * masked_position_variance_regularization(
                canonical_prediction, full_mask
            )
        return WindowJEPAOutput(
            loss=loss,
            masked_loss=masked_loss,
            canonical_loss=canonical_loss,
            dense_loss=masked_loss,
            visible_loss=masked_loss.new_zeros(()),
            deep_supervision_loss=masked_loss.new_zeros(()),
            prediction=prediction,
            target=target_tokens,
            prediction_std=masked_position_standard_deviation(
                prediction.detach(), target_mask
            ),
            target_std=masked_position_standard_deviation(target_tokens, target_mask),
        )

    def feedforward_sequence(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        context_duration_ms: torch.Tensor,
        target_duration_ms: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        dense: bool,
    ) -> WindowJEPAOutput:
        """Evaluate supervised clip frames independently in one flat forward pass.

        Sequence loading and temporal state are deliberately independent here: burn-in
        frames are removed with ``loss_mask`` and no recurrent state or TBPTT boundary
        is consumed. The sequence-shaped prediction and EMA-target views are retained
        for temporal diagnostics and future loss wiring; they are not SIGReg projector
        latents by themselves.
        """

        if isinstance(self.online_encoder, RecurrentVJEPA21EventVisionTransformer):
            raise ValueError("feedforward sequence objectives require a non-recurrent encoder")
        if x_context.ndim != 5 or x_context.shape != x_target.shape:
            raise ValueError("feedforward sequence inputs must share shape [B,T,C,H,W]")
        batch_size, steps = x_context.shape[:2]
        if context_duration_ms.shape != (batch_size, steps) or (
            target_duration_ms.shape != (batch_size, steps)
        ):
            raise ValueError("feedforward sequence durations must have shape [B,T]")
        expected_mask_shape = (batch_size, steps, self.num_patches)
        if context_mask.shape != expected_mask_shape or target_mask.shape != expected_mask_shape:
            raise ValueError("feedforward sequence patch masks must have shape [B,T,N]")
        if loss_mask.dtype != torch.bool or loss_mask.shape != (batch_size, steps):
            raise ValueError("sequence_loss_mask must be boolean with shape [B,T]")
        if not bool((loss_mask == loss_mask[:1]).all()):
            raise ValueError("all clips in a batch must share sequence_loss_mask")
        supervised_indices = torch.nonzero(
            loss_mask[0], as_tuple=False
        ).flatten()
        supervised_steps = int(supervised_indices.numel())
        if supervised_steps == 0:
            raise ValueError("sequence clip has no loss-bearing timesteps")

        # Select before flattening so burn-in windows never enter the encoder. The
        # resulting order is batch-major and reshapes losslessly back to [B,T,N,D].
        def flatten_frames(value: torch.Tensor) -> torch.Tensor:
            selected = value.index_select(1, supervised_indices)
            return selected.flatten(0, 1)

        arguments = {
            "x_context": flatten_frames(x_context),
            "x_target": flatten_frames(x_target),
            "dt_context_ms": flatten_frames(context_duration_ms),
            "dt_target_ms": flatten_frames(target_duration_ms),
            "context_mask": flatten_frames(context_mask),
            "target_mask": flatten_frames(target_mask),
        }
        output = (
            self.dense_window_jepa(**arguments)
            if dense
            else self.forward(**arguments, objective="window_jepa")
        )
        sequence_shape = (
            batch_size,
            supervised_steps,
            self.num_patches,
            output.prediction.shape[-1],
        )
        return replace(
            output,
            prediction_sequence=output.prediction.reshape(sequence_shape),
            target_sequence=output.target.reshape(sequence_shape),
        )

    def _recurrent_encoders(
        self,
    ) -> tuple[
        RecurrentVJEPA21EventVisionTransformer,
        RecurrentVJEPA21EventVisionTransformer,
    ]:
        if not isinstance(
            self.online_encoder, RecurrentVJEPA21EventVisionTransformer
        ) or not isinstance(
            self.target_encoder, RecurrentVJEPA21EventVisionTransformer
        ):
            raise ValueError("recurrent objectives require recurrent V-JEPA 2.1 encoders")
        return self.online_encoder, self.target_encoder

    @torch.no_grad()
    def recurrent_burn_in(
        self,
        x: torch.Tensor,
        duration_ms: torch.Tensor,
        context_mask: torch.Tensor | None,
        online_state: RecurrentState | None = None,
    ) -> RecurrentState | None:
        """Initialize online memory without creating a gradient graph or target state."""

        online_encoder, _ = self._recurrent_encoders()
        if x.ndim != 5:
            raise ValueError("recurrent burn-in input must have shape [B,T,C,H,W]")
        batch_size, steps = x.shape[:2]
        if duration_ms.shape != (batch_size, steps):
            raise ValueError("burn-in duration must have shape [B,T]")
        if context_mask is not None and context_mask.shape != (
            batch_size,
            steps,
            self.num_patches,
        ):
            raise ValueError("burn-in context mask must have shape [B,T,N]")
        state = online_state
        for index in range(steps):
            scale = self.scale_embedding(duration_ms[:, index])
            if not self.condition_on_scale:
                scale = torch.zeros_like(scale)
            step_mask = (
                None
                if online_encoder.recurrent_placement == "post_encoder"
                or context_mask is None
                else context_mask[:, index]
            )
            _, state = online_encoder.forward_recurrent(
                x[:, index],
                scale,
                step_mask,
                state=state,
            )
        return detach_recurrent_state(state)

    def recurrent_sequence(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        context_duration_ms: torch.Tensor,
        target_duration_ms: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        online_state: RecurrentState | None,
        *,
        dense: bool,
    ) -> WindowJEPAOutput:
        """Unroll one BPTT chunk while resetting the EMA target state each step."""

        self._recurrent_encoders()
        if x_context.ndim != 5 or x_context.shape != x_target.shape:
            raise ValueError("recurrent input must have shape [B,T,C,H,W]")
        batch_size, steps = x_context.shape[:2]
        if steps <= 0:
            raise ValueError("a recurrent chunk must contain at least one timestep")
        if context_duration_ms.shape != (batch_size, steps) or (
            target_duration_ms.shape != (batch_size, steps)
        ):
            raise ValueError("recurrent durations must have shape [B,T]")
        expected_mask_shape = (batch_size, steps, self.num_patches)
        if context_mask.shape != expected_mask_shape or target_mask.shape != expected_mask_shape:
            raise ValueError("recurrent masks must have shape [B,T,N]")

        state = online_state
        outputs: list[WindowJEPAOutput] = []
        for index in range(steps):
            if dense:
                output = self._recurrent_dense_step(
                    x_context=x_context[:, index],
                    x_target=x_target[:, index],
                    context_duration_ms=context_duration_ms[:, index],
                    target_duration_ms=target_duration_ms[:, index],
                    context_mask=context_mask[:, index],
                    target_mask=target_mask[:, index],
                    online_state=state,
                )
            else:
                output = self._recurrent_step(
                    x_context=x_context[:, index],
                    x_target=x_target[:, index],
                    context_duration_ms=context_duration_ms[:, index],
                    target_duration_ms=target_duration_ms[:, index],
                    context_mask=context_mask[:, index],
                    target_mask=target_mask[:, index],
                    online_state=state,
                )
            state = output.online_state
            outputs.append(output)

        prediction = torch.cat([output.prediction for output in outputs], dim=0)
        target = torch.cat([output.target for output in outputs], dim=0)
        flattened_target_mask = target_mask.transpose(0, 1).reshape(
            steps * batch_size, self.num_patches
        )

        def mean(name: str) -> torch.Tensor:
            return torch.stack([getattr(output, name) for output in outputs]).mean()

        return WindowJEPAOutput(
            loss=mean("loss"),
            masked_loss=mean("masked_loss"),
            canonical_loss=mean("canonical_loss"),
            dense_loss=mean("dense_loss"),
            visible_loss=mean("visible_loss"),
            deep_supervision_loss=mean("deep_supervision_loss"),
            prediction=prediction,
            target=target,
            prediction_std=masked_position_standard_deviation(
                prediction.detach(), flattened_target_mask
            ),
            target_std=masked_position_standard_deviation(
                target, flattened_target_mask
            ),
            online_state=state,
        )

    def _event_support_pool(
        self,
        tokens: torch.Tensor,
        event_activity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool one vector per sample without letting event-rich clips dominate."""

        if tokens.ndim != 3:
            raise ValueError("event-support pooling expects tokens [B,N,D]")
        if event_activity.shape != tokens.shape[:2]:
            raise ValueError("event activity must have shape [B,N]")
        if event_activity.device != tokens.device:
            raise ValueError("event activity and tokens must share a device")
        active = event_activity >= self.future_active_min_events
        weights = torch.where(
            active,
            torch.ones_like(event_activity, dtype=torch.float32),
            torch.full_like(
                event_activity,
                self.future_activity_floor,
                dtype=torch.float32,
            ),
        )
        normalized = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (tokens.float() * normalized.unsqueeze(-1)).sum(dim=1)
        # A threshold greater than one must not discard a non-empty frame just
        # because every patch falls below that threshold. Thresholding controls
        # the pooling weights; raw event presence controls SIGReg validity.
        return pooled, event_activity.sum(dim=1) > 0

    def _event_support_contrast(
        self,
        tokens: torch.Tensor,
        event_activity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Contrast active and inactive frame regions to reject spatial collapse."""

        if tokens.ndim != 3 or event_activity.shape != tokens.shape[:2]:
            raise ValueError(
                "event-support contrast expects tokens [B,N,D] and activity [B,N]"
            )
        if event_activity.device != tokens.device:
            raise ValueError("event activity and tokens must share a device")
        active = event_activity >= self.future_active_min_events
        inactive = ~active
        # A threshold can place every patch in the same class, especially for a
        # dense frame or when active_min_events is greater than one. Where raw
        # event counts genuinely vary, recover an event-derived contrast by
        # splitting the lower/higher count halves. Equal-count rows remain
        # invalid: an index-based tie break would turn this into a positional
        # shortcut rather than an event-support regularizer.
        if tokens.shape[1] >= 2:
            has_events = event_activity.sum(dim=1) > 0
            single_class = active.all(dim=1) | inactive.all(dim=1)
            has_count_variation = event_activity.amax(dim=1) > event_activity.amin(
                dim=1
            )
            use_count_split = single_class & has_events & has_count_variation
            half = tokens.shape[1] // 2
            order = event_activity.argsort(dim=1)
            low_activity = torch.zeros_like(active)
            high_activity = torch.zeros_like(active)
            low_activity.scatter_(1, order[:, :half], True)
            high_activity.scatter_(1, order[:, half:], True)
            active = torch.where(
                use_count_split.unsqueeze(1), high_activity, active
            )
            inactive = torch.where(
                use_count_split.unsqueeze(1), low_activity, inactive
            )
        active_count = active.sum(dim=1, keepdim=True)
        inactive_count = inactive.sum(dim=1, keepdim=True)
        active_mean = (
            tokens.float() * active.unsqueeze(-1).to(torch.float32)
        ).sum(dim=1) / active_count.clamp_min(1).to(torch.float32)
        inactive_mean = (
            tokens.float() * inactive.unsqueeze(-1).to(torch.float32)
        ).sum(dim=1) / inactive_count.clamp_min(1).to(torch.float32)
        valid = (active_count[:, 0] > 0) & (inactive_count[:, 0] > 0)
        return active_mean - inactive_mean, valid

    @staticmethod
    def _previous_hidden_tokens(
        state: RecurrentState | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if state is None:
            return torch.zeros_like(reference)
        hidden = state[0] if isinstance(state, tuple) else state
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 4:
            raise TypeError("recurrent state must contain a hidden grid [B,D,H,W]")
        tokens = hidden.flatten(2).transpose(1, 2)
        if tokens.shape != reference.shape:
            raise ValueError("recurrent hidden grid and token grid do not match")
        return tokens

    @staticmethod
    @torch.no_grad()
    def _global_fixed_position_standard_deviations(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Measure collapse across the global batch at every fixed patch position."""

        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("prediction and target must share shape [B,N,D]")
        prediction_fp32 = prediction.detach().float()
        target_fp32 = target.detach().float()
        moments = torch.stack(
            (
                prediction_fp32.sum(dim=0),
                prediction_fp32.square().sum(dim=0),
                target_fp32.sum(dim=0),
                target_fp32.square().sum(dim=0),
            )
        )
        count = prediction_fp32.new_tensor(float(prediction.shape[0]))
        if (
            distributed.is_available()
            and distributed.is_initialized()
            and distributed.get_world_size() > 1
        ):
            distributed.all_reduce(moments, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(count, op=distributed.ReduceOp.SUM)
        denominator = count.clamp_min(1.0)
        prediction_mean = moments[0] / denominator
        target_mean = moments[2] / denominator
        prediction_variance = (
            moments[1] / denominator - prediction_mean.square()
        ).clamp_min(0.0)
        target_variance = (
            moments[3] / denominator - target_mean.square()
        ).clamp_min(0.0)
        return prediction_variance.sqrt().mean(), target_variance.sqrt().mean()

    def recurrent_future_sequence(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        context_duration_ms: torch.Tensor,
        target_duration_ms: torch.Tensor,
        context_event_activity: torch.Tensor,
        target_event_activity: torch.Tensor,
        online_state: RecurrentState | None,
    ) -> WindowJEPAOutput:
        """Predict full future EMA frame latents from causal post-ViT memory.

        Spatial masks are deliberately absent from this path. Both the online
        frame encoder and the stateless EMA teacher see full event windows; the
        event-support map is applied only to loss balancing and SIGReg pooling.
        """

        online_encoder, target_encoder = self._recurrent_encoders()
        if online_encoder.recurrent_placement != "post_encoder":
            raise ValueError(
                "recurrent_future_jepa requires post_encoder recurrence"
            )
        if x_context.ndim != 5 or x_context.shape != x_target.shape:
            raise ValueError("future recurrent inputs must share shape [B,T,C,H,W]")
        batch_size, steps = x_context.shape[:2]
        if steps <= 0:
            raise ValueError("a future recurrent chunk requires at least one timestep")
        if context_duration_ms.shape != (batch_size, steps) or (
            target_duration_ms.shape != (batch_size, steps)
        ):
            raise ValueError("future recurrent durations must have shape [B,T]")
        activity_shape = (batch_size, steps, self.num_patches)
        if context_event_activity.shape != activity_shape or (
            target_event_activity.shape != activity_shape
        ):
            raise ValueError("future event activity must have shape [B,T,N]")

        state = online_state
        outputs: list[WindowJEPAOutput] = []
        for index in range(steps):
            context_ms = context_duration_ms[:, index].reshape(batch_size)
            target_ms = target_duration_ms[:, index].reshape(batch_size)
            source_scale, target_scale, ratio_scale = self._scale_features(
                context_ms, target_ms
            )
            previous_state = state
            frame_tokens, recurrent_tokens, state = (
                online_encoder.forward_frame_and_recurrent(
                    x_context[:, index],
                    source_scale,
                    None,
                    state=state,
                )
            )
            with torch.no_grad():
                ema_scale = self.target_scale_embedding(target_ms)
                if not self.condition_on_scale:
                    ema_scale = torch.zeros_like(ema_scale)
                target_tokens = target_encoder.forward_frame(
                    x_target[:, index], ema_scale, None
                )

            full_mask = torch.ones(
                (batch_size, self.num_patches),
                dtype=torch.bool,
                device=x_context.device,
            )
            prediction = self.predictor(
                context_tokens=recurrent_tokens,
                context_keep_mask=full_mask,
                target_mask=full_mask,
                source_scale=source_scale,
                target_scale=target_scale,
                ratio_scale=ratio_scale,
            )
            balanced = balanced_event_support_latent_prediction_loss(
                prediction,
                target_tokens,
                target_event_activity[:, index],
                active_threshold=float(self.future_active_min_events - 1),
                kind=self.loss_kind,
            )

            zero = balanced.loss.new_zeros(())
            frame_sigreg = zero
            support_sigreg = zero
            temporal_sigreg = zero
            frame_samples = zero
            support_samples = zero
            temporal_samples = zero
            frame_real = zero
            frame_imaginary = zero
            support_real = zero
            support_imaginary = zero
            temporal_real = zero
            temporal_imaginary = zero
            context_activity = context_event_activity[:, index]
            if "frame" in self.future_regularizers:
                frame_vector, frame_valid = self._event_support_pool(
                    frame_tokens, context_activity
                )
                frame_output = self.future_regularizers["frame"](
                    frame_vector, valid_mask=frame_valid
                )
                frame_sigreg = frame_output.loss
                frame_samples = frame_output.effective_samples
                frame_real = frame_output.real_error
                frame_imaginary = frame_output.imaginary_error
                support_vector, support_valid = self._event_support_contrast(
                    frame_tokens, context_activity
                )
                support_output = self.future_regularizers["support"](
                    support_vector, valid_mask=support_valid
                )
                support_sigreg = support_output.loss
                support_samples = support_output.effective_samples
                support_real = support_output.real_error
                support_imaginary = support_output.imaginary_error
            if "temporal" in self.future_regularizers:
                previous_tokens = self._previous_hidden_tokens(
                    previous_state, recurrent_tokens
                )
                temporal_vector, temporal_valid = self._event_support_pool(
                    recurrent_tokens - previous_tokens,
                    context_activity,
                )
                temporal_output = self.future_regularizers["temporal"](
                    temporal_vector, valid_mask=temporal_valid
                )
                temporal_sigreg = temporal_output.loss
                temporal_samples = temporal_output.effective_samples
                temporal_real = temporal_output.real_error
                temporal_imaginary = temporal_output.imaginary_error
            support_available = (
                support_samples >= 2
            ).to(dtype=frame_sigreg.dtype)
            frame_regularization = (
                frame_sigreg + support_sigreg
            ) / (1.0 + support_available)
            sigreg_loss = (
                self.frame_sigreg_weight * frame_regularization
                + self.temporal_sigreg_weight * temporal_sigreg
            )
            loss = balanced.loss + sigreg_loss
            prediction_std, target_std = (
                self._global_fixed_position_standard_deviations(
                    prediction,
                    target_tokens,
                )
            )
            outputs.append(
                WindowJEPAOutput(
                    loss=loss,
                    masked_loss=balanced.loss,
                    canonical_loss=zero,
                    dense_loss=balanced.loss,
                    visible_loss=zero,
                    deep_supervision_loss=zero,
                    prediction=prediction,
                    target=target_tokens,
                    prediction_std=prediction_std,
                    target_std=target_std,
                    online_state=state,
                    future_prediction_loss=balanced.loss,
                    active_prediction_loss=balanced.active_loss,
                    inactive_prediction_loss=balanced.inactive_loss,
                    frame_sigreg_loss=frame_sigreg,
                    support_sigreg_loss=support_sigreg,
                    temporal_sigreg_loss=temporal_sigreg,
                    sigreg_loss=sigreg_loss,
                    active_patch_fraction=(
                        target_event_activity[:, index]
                        >= self.future_active_min_events
                    ).float().mean(),
                    context_active_patch_fraction=(
                        context_activity >= self.future_active_min_events
                    ).float().mean(),
                    frame_sigreg_samples=frame_samples,
                    support_sigreg_samples=support_samples,
                    temporal_sigreg_samples=temporal_samples,
                    frame_sigreg_real_error=frame_real,
                    frame_sigreg_imaginary_error=frame_imaginary,
                    support_sigreg_real_error=support_real,
                    support_sigreg_imaginary_error=support_imaginary,
                    temporal_sigreg_real_error=temporal_real,
                    temporal_sigreg_imaginary_error=temporal_imaginary,
                    active_prediction_sum=(
                        balanced.active_loss
                        * balanced.active_sample_count.to(
                            balanced.active_loss.dtype
                        )
                    ),
                    active_prediction_count=balanced.active_sample_count.float(),
                    inactive_prediction_sum=(
                        balanced.inactive_loss
                        * balanced.inactive_sample_count.to(
                            balanced.inactive_loss.dtype
                        )
                    ),
                    inactive_prediction_count=(
                        balanced.inactive_sample_count.float()
                    ),
                )
            )

        prediction = torch.cat([output.prediction for output in outputs], dim=0)
        target = torch.cat([output.target for output in outputs], dim=0)
        def mean(name: str) -> torch.Tensor:
            values = [getattr(output, name) for output in outputs]
            if any(value is None for value in values):
                raise RuntimeError(f"future objective metric {name} is missing")
            return torch.stack(values).mean()  # type: ignore[arg-type]

        active_sum = mean("active_prediction_sum")
        active_count = mean("active_prediction_count")
        inactive_sum = mean("inactive_prediction_sum")
        inactive_count = mean("inactive_prediction_count")

        return WindowJEPAOutput(
            loss=mean("loss"),
            masked_loss=mean("masked_loss"),
            canonical_loss=mean("canonical_loss"),
            dense_loss=mean("dense_loss"),
            visible_loss=mean("visible_loss"),
            deep_supervision_loss=mean("deep_supervision_loss"),
            prediction=prediction,
            target=target,
            prediction_std=mean("prediction_std"),
            target_std=mean("target_std"),
            online_state=state,
            future_prediction_loss=mean("future_prediction_loss"),
            active_prediction_loss=active_sum / active_count.clamp_min(1.0),
            inactive_prediction_loss=(
                inactive_sum / inactive_count.clamp_min(1.0)
            ),
            frame_sigreg_loss=mean("frame_sigreg_loss"),
            support_sigreg_loss=mean("support_sigreg_loss"),
            temporal_sigreg_loss=mean("temporal_sigreg_loss"),
            sigreg_loss=mean("sigreg_loss"),
            active_patch_fraction=mean("active_patch_fraction"),
            context_active_patch_fraction=mean(
                "context_active_patch_fraction"
            ),
            frame_sigreg_samples=mean("frame_sigreg_samples"),
            support_sigreg_samples=mean("support_sigreg_samples"),
            temporal_sigreg_samples=mean("temporal_sigreg_samples"),
            frame_sigreg_real_error=mean("frame_sigreg_real_error"),
            frame_sigreg_imaginary_error=mean(
                "frame_sigreg_imaginary_error"
            ),
            support_sigreg_real_error=mean("support_sigreg_real_error"),
            support_sigreg_imaginary_error=mean(
                "support_sigreg_imaginary_error"
            ),
            temporal_sigreg_real_error=mean("temporal_sigreg_real_error"),
            temporal_sigreg_imaginary_error=mean(
                "temporal_sigreg_imaginary_error"
            ),
            active_prediction_sum=active_sum,
            active_prediction_count=active_count,
            inactive_prediction_sum=inactive_sum,
            inactive_prediction_count=inactive_count,
        )

    def _recurrent_step(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        context_duration_ms: torch.Tensor,
        target_duration_ms: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        online_state: RecurrentState | None,
    ) -> WindowJEPAOutput:
        online_encoder, target_encoder = self._recurrent_encoders()
        batch_size = x_context.shape[0]
        context_duration_ms = context_duration_ms.reshape(batch_size)
        target_duration_ms = target_duration_ms.reshape(batch_size)
        source_scale, target_scale, ratio_scale = self._scale_features(
            context_duration_ms, target_duration_ms
        )
        context_tokens, next_state = online_encoder.forward_recurrent(
            x_context,
            source_scale,
            context_mask,
            state=online_state,
        )
        with torch.no_grad():
            ema_scale = self.target_scale_embedding(target_duration_ms)
            if not self.condition_on_scale:
                ema_scale = torch.zeros_like(ema_scale)
            # R0 deliberately defines the target as the current full 50 ms
            # window.  It never carries teacher memory across timesteps.
            target_tokens, _ = target_encoder.forward_recurrent(
                x_target,
                ema_scale,
                None,
                state=None,
            )
        prediction = self.predictor(
            context_tokens=context_tokens,
            context_keep_mask=context_mask,
            target_mask=target_mask,
            source_scale=source_scale,
            target_scale=target_scale,
            ratio_scale=ratio_scale,
        )
        masked_loss = latent_prediction_loss(
            prediction, target_tokens, target_mask=target_mask, kind=self.loss_kind
        )
        zero = masked_loss.new_zeros(())
        return WindowJEPAOutput(
            loss=masked_loss,
            masked_loss=masked_loss,
            canonical_loss=zero,
            dense_loss=masked_loss,
            visible_loss=zero,
            deep_supervision_loss=zero,
            prediction=prediction,
            target=target_tokens,
            prediction_std=masked_position_standard_deviation(
                prediction.detach(), target_mask
            ),
            target_std=masked_position_standard_deviation(target_tokens, target_mask),
            online_state=next_state,
        )

    def _recurrent_dense_step(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        context_duration_ms: torch.Tensor,
        target_duration_ms: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        online_state: RecurrentState | None,
    ) -> WindowJEPAOutput:
        online_encoder, target_encoder = self._recurrent_encoders()
        batch_size = x_context.shape[0]
        context_duration_ms = context_duration_ms.reshape(batch_size)
        target_duration_ms = target_duration_ms.reshape(batch_size)
        source_scale, target_scale, ratio_scale = self._scale_features(
            context_duration_ms, target_duration_ms
        )
        context_layers, next_state = online_encoder.forward_recurrent_intermediates(
            x_context,
            source_scale,
            context_mask,
            state=online_state,
        )
        with torch.no_grad():
            ema_scale = self.target_scale_embedding(target_duration_ms)
            if not self.condition_on_scale:
                ema_scale = torch.zeros_like(ema_scale)
            target_layers, _ = target_encoder.forward_recurrent_intermediates(
                x_target,
                ema_scale,
                None,
                state=None,
            )
        if len(context_layers) != len(target_layers) or not context_layers:
            raise RuntimeError("recurrent deep-supervision outputs do not match")

        full_mask = torch.ones_like(context_mask)
        predictions: list[torch.Tensor] = []
        dense_losses: list[torch.Tensor] = []
        masked_losses: list[torch.Tensor] = []
        visible_losses: list[torch.Tensor] = []
        for context_tokens, target_tokens in zip(
            context_layers, target_layers, strict=True
        ):
            prediction = self.predictor(
                context_tokens=context_tokens,
                context_keep_mask=context_mask,
                target_mask=full_mask,
                source_scale=source_scale,
                target_scale=target_scale,
                ratio_scale=ratio_scale,
            )
            predictions.append(prediction)
            masked_layer_loss = latent_prediction_loss(
                prediction, target_tokens, target_mask=target_mask, kind=self.loss_kind
            )
            visible_layer_loss = latent_prediction_loss(
                prediction, target_tokens, target_mask=context_mask, kind=self.loss_kind
            )
            masked_losses.append(masked_layer_loss)
            visible_losses.append(visible_layer_loss)
            dense_losses.append((masked_layer_loss + visible_layer_loss) * 0.5)

        dense_loss = torch.stack(dense_losses).mean()
        masked_loss = torch.stack(masked_losses).mean()
        visible_loss = torch.stack(visible_losses).mean()
        deep_supervision_loss = (
            torch.stack(dense_losses[:-1]).mean()
            if len(dense_losses) > 1
            else dense_loss.new_zeros(())
        )
        prediction = predictions[-1]
        target = target_layers[-1]
        return WindowJEPAOutput(
            loss=dense_loss,
            masked_loss=masked_loss,
            canonical_loss=dense_loss.new_zeros(()),
            dense_loss=dense_loss,
            visible_loss=visible_loss,
            deep_supervision_loss=deep_supervision_loss,
            prediction=prediction,
            target=target,
            prediction_std=masked_position_standard_deviation(
                prediction.detach(), target_mask
            ),
            target_std=masked_position_standard_deviation(target, target_mask),
            online_state=next_state,
        )

    def dense_window_jepa(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        dt_context_ms: torch.Tensor,
        dt_target_ms: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> WindowJEPAOutput:
        """V-JEPA 2.1-style all-token loss with intermediate-layer supervision.

        The online encoder sees only retained context patches. The predictor
        queries the complete grid, and the objective applies equal-weight loss
        to masked target tokens and retained visible tokens. Every selected
        encoder depth uses the same flat spatial grid and shared predictor.
        """

        if not isinstance(self.online_encoder, VJEPA21EventVisionTransformer):
            raise ValueError("dense_window_jepa requires a V-JEPA 2.1 encoder")
        if x_context.shape != x_target.shape or x_context.ndim != 4:
            raise ValueError("context and target inputs must share shape [B, C, H, W]")
        batch_size = x_context.shape[0]
        context_ms = dt_context_ms.reshape(batch_size)
        target_ms = dt_target_ms.reshape(batch_size)
        source_scale, target_scale, ratio_scale = self._scale_features(context_ms, target_ms)
        context_layers = self.online_encoder.forward_intermediates(
            x_context, source_scale, context_mask
        )
        with torch.no_grad():
            ema_target_scale = self.target_scale_embedding(target_ms)
            if not self.condition_on_scale:
                ema_target_scale = torch.zeros_like(ema_target_scale)
            if not isinstance(self.target_encoder, VJEPA21EventVisionTransformer):
                raise RuntimeError("online and EMA target encoder architectures differ")
            target_layers = self.target_encoder.forward_intermediates(
                x_target, ema_target_scale, None
            )
        if len(context_layers) != len(target_layers) or not context_layers:
            raise RuntimeError("deep-supervision encoder outputs do not match")

        full_mask = torch.ones_like(context_mask)
        predictions: list[torch.Tensor] = []
        dense_losses: list[torch.Tensor] = []
        masked_losses: list[torch.Tensor] = []
        visible_losses: list[torch.Tensor] = []
        for context_tokens, target_tokens in zip(
            context_layers, target_layers, strict=True
        ):
            prediction = self.predictor(
                context_tokens=context_tokens,
                context_keep_mask=context_mask,
                target_mask=full_mask,
                source_scale=source_scale,
                target_scale=target_scale,
                ratio_scale=ratio_scale,
            )
            predictions.append(prediction)
            masked_layer_loss = latent_prediction_loss(
                prediction, target_tokens, target_mask=target_mask, kind=self.loss_kind
            )
            visible_layer_loss = latent_prediction_loss(
                prediction, target_tokens, target_mask=context_mask, kind=self.loss_kind
            )
            masked_losses.append(masked_layer_loss)
            visible_losses.append(visible_layer_loss)
            dense_losses.append((masked_layer_loss + visible_layer_loss) * 0.5)

        dense_loss = torch.stack(dense_losses).mean()
        masked_loss = torch.stack(masked_losses).mean()
        visible_loss = torch.stack(visible_losses).mean()
        deep_supervision_loss = (
            torch.stack(dense_losses[:-1]).mean()
            if len(dense_losses) > 1
            else dense_loss.new_zeros(())
        )
        prediction = predictions[-1]
        target_tokens = target_layers[-1]

        canonical_loss = dense_loss.new_zeros(())
        canonical_prediction = prediction
        if self.canonical_query_weight:
            full_context_tokens = self.online_encoder(x_context, source_scale, None)
            canonical_prediction = self.predictor(
                context_tokens=full_context_tokens,
                context_keep_mask=full_mask,
                target_mask=full_mask,
                source_scale=source_scale,
                target_scale=target_scale,
                ratio_scale=ratio_scale,
            )
            canonical_loss = latent_prediction_loss(
                canonical_prediction, target_tokens, target_mask=None, kind=self.loss_kind
            )
        loss = dense_loss + self.canonical_query_weight * canonical_loss
        if self.variance_weight:
            loss = loss + self.variance_weight * masked_position_variance_regularization(
                canonical_prediction, full_mask
            )
        return WindowJEPAOutput(
            loss=loss,
            masked_loss=masked_loss,
            canonical_loss=canonical_loss,
            dense_loss=dense_loss,
            visible_loss=visible_loss,
            deep_supervision_loss=deep_supervision_loss,
            prediction=prediction,
            target=target_tokens,
            prediction_std=masked_position_standard_deviation(
                prediction.detach(), target_mask
            ),
            target_std=masked_position_standard_deviation(target_tokens, target_mask),
        )

    def feature_consistency(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        dt_context_ms: torch.Tensor,
        dt_target_ms: torch.Tensor,
    ) -> WindowJEPAOutput:
        """B4 baseline: direct global-feature agreement across windows."""

        batch_size = x_context.shape[0]
        source_scale, target_scale, _ = self._scale_features(
            dt_context_ms.reshape(batch_size), dt_target_ms.reshape(batch_size)
        )
        online_tokens = self.online_encoder(x_context, source_scale, None)
        with torch.no_grad():
            ema_target_scale = self.target_scale_embedding(dt_target_ms.reshape(batch_size))
            if not self.condition_on_scale:
                ema_target_scale = torch.zeros_like(ema_target_scale)
            target_tokens = self.target_encoder(x_target, ema_target_scale, None)
        prediction = online_tokens.mean(dim=1, keepdim=True)
        target = target_tokens.mean(dim=1, keepdim=True)
        loss = functional.smooth_l1_loss(
            functional.layer_norm(prediction, (prediction.shape[-1],)),
            functional.layer_norm(target.detach(), (target.shape[-1],)),
        )
        if self.variance_weight:
            loss = loss + self.variance_weight * variance_regularization(prediction)
        if self.covariance_weight:
            loss = loss + self.covariance_weight * covariance_regularization(prediction)
        return WindowJEPAOutput(
            loss=loss,
            masked_loss=loss,
            canonical_loss=loss.new_zeros(()),
            dense_loss=loss,
            visible_loss=loss.new_zeros(()),
            deep_supervision_loss=loss.new_zeros(()),
            prediction=prediction,
            target=target,
            prediction_std=feature_standard_deviation(prediction.detach()),
            target_std=feature_standard_deviation(target),
        )

    def encode_only(self, x: torch.Tensor, duration_ms: torch.Tensor) -> torch.Tensor:
        duration_ms = duration_ms.reshape(x.shape[0])
        scale = self.scale_embedding(duration_ms)
        if not self.condition_on_scale:
            scale = torch.zeros_like(scale)
        return self.online_encoder(x, scale, None)

    def encode_recurrent(
        self,
        x: torch.Tensor,
        duration_ms: torch.Tensor,
        online_state: RecurrentState | None = None,
        *,
        detach_state: bool = True,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Encode one causal downstream step and explicitly return its memory."""

        online_encoder, _ = self._recurrent_encoders()
        duration_ms = duration_ms.reshape(x.shape[0])
        scale = self.scale_embedding(duration_ms)
        if not self.condition_on_scale:
            scale = torch.zeros_like(scale)
        return online_encoder.forward_recurrent(
            x,
            scale,
            None,
            state=online_state,
            detach_state=detach_state,
        )

    def canonical_latent(
        self,
        x: torch.Tensor,
        source_duration_ms: torch.Tensor,
        canonical_duration_ms: float | torch.Tensor,
    ) -> torch.Tensor:
        """Convert any input window into a full canonical patch-token grid."""

        batch_size = x.shape[0]
        source_duration_ms = source_duration_ms.reshape(batch_size)
        canonical = torch.as_tensor(
            canonical_duration_ms,
            device=x.device,
            dtype=source_duration_ms.dtype,
        )
        if canonical.ndim == 0:
            canonical = canonical.expand(batch_size)
        canonical = canonical.reshape(batch_size)
        source_scale, target_scale, ratio_scale = self._scale_features(
            source_duration_ms, canonical
        )
        full_mask = torch.ones(
            (batch_size, self.num_patches), dtype=torch.bool, device=x.device
        )
        context_tokens = self.online_encoder(x, source_scale, None)
        return self.predictor(
            context_tokens=context_tokens,
            context_keep_mask=full_mask,
            target_mask=full_mask,
            source_scale=source_scale,
            target_scale=target_scale,
            ratio_scale=ratio_scale,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        update_ema(self.online_encoder, self.target_encoder, momentum)
        update_ema(self.scale_embedding, self.target_scale_embedding, momentum)
