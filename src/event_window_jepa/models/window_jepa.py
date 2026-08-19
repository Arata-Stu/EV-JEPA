from __future__ import annotations

from dataclasses import dataclass

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


class WindowJEPA(nn.Module):
    def __init__(
        self,
        encoder: EventVisionTransformer | VJEPA21EventVisionTransformer,
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
    ) -> WindowJEPAOutput:
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
