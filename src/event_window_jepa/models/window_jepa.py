from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as functional
from torch import nn

from event_window_jepa.losses.latent_prediction import latent_prediction_loss
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

    @property
    def num_patches(self) -> int:
        return self.online_encoder.num_patches

    def train(self, mode: bool = True) -> WindowJEPA:
        super().train(mode)
        self.target_encoder.eval()
        self.target_scale_embedding.eval()
        return self

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
        context_mask: torch.Tensor,
        online_state: RecurrentState | None = None,
    ) -> RecurrentState | None:
        """Initialize online memory without creating a gradient graph or target state."""

        online_encoder, _ = self._recurrent_encoders()
        if x.ndim != 5:
            raise ValueError("recurrent burn-in input must have shape [B,T,C,H,W]")
        batch_size, steps = x.shape[:2]
        if duration_ms.shape != (batch_size, steps):
            raise ValueError("burn-in duration must have shape [B,T]")
        if context_mask.shape != (batch_size, steps, self.num_patches):
            raise ValueError("burn-in context mask must have shape [B,T,N]")
        state = online_state
        for index in range(steps):
            scale = self.scale_embedding(duration_ms[:, index])
            if not self.condition_on_scale:
                scale = torch.zeros_like(scale)
            _, state = online_encoder.forward_recurrent(
                x[:, index],
                scale,
                context_mask[:, index],
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
