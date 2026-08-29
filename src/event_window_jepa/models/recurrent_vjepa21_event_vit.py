from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from event_window_jepa.models.token_utils import gather_equal_count, validate_patch_mask
from event_window_jepa.models.vjepa21_event_vit import VJEPA21EventVisionTransformer


ConvLSTMState = tuple[torch.Tensor, torch.Tensor]
RecurrentState = torch.Tensor | ConvLSTMState
RecurrentCellKind = Literal["conv_lstm", "conv_gru", "convlstm", "convgru"]
RecurrentPlacement = Literal["pre_encoder", "post_encoder"]


def _validate_kernel_size(kernel_size: int) -> None:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("recurrent kernel_size must be a positive odd integer")


def _validate_state_tensor(
    state: torch.Tensor,
    reference: torch.Tensor,
    hidden_channels: int,
    name: str,
) -> None:
    expected_shape = (
        reference.shape[0],
        hidden_channels,
        reference.shape[-2],
        reference.shape[-1],
    )
    if tuple(state.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(state.shape)}")
    if state.device != reference.device:
        raise ValueError(
            f"{name} must be on {reference.device}, got {state.device}"
        )
    if state.dtype != reference.dtype:
        raise ValueError(f"{name} must have dtype {reference.dtype}, got {state.dtype}")


def detach_recurrent_state(state: RecurrentState | None) -> RecurrentState | None:
    """Detach an externally owned state at a truncated-BPTT boundary."""

    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple) and len(state) == 2 and all(
        isinstance(value, torch.Tensor) for value in state
    ):
        return state[0].detach(), state[1].detach()
    raise TypeError("recurrent state must be a tensor or an (hidden, cell) tensor pair")


def _validate_state_reset_tensor(
    state: torch.Tensor,
    reset_mask: torch.Tensor,
    name: str,
) -> None:
    if state.ndim != 4:
        raise ValueError(f"{name} must have shape [B,C,H,W], got {tuple(state.shape)}")
    if not state.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype, got {state.dtype}")
    expected_mask_shape = (state.shape[0],)
    if tuple(reset_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"reset_mask must have shape {expected_mask_shape}, got {tuple(reset_mask.shape)}"
        )
    if reset_mask.device != state.device:
        raise ValueError(
            f"reset_mask must be on {state.device} with {name}, got {reset_mask.device}"
        )


def _zero_reset_state_tensor(
    state: torch.Tensor,
    reset_mask: torch.Tensor,
) -> torch.Tensor:
    broadcast_mask = reset_mask.reshape(
        reset_mask.shape[0], *((1,) * (state.ndim - 1))
    )
    return state.masked_fill(broadcast_mask, 0.0)


def reset_recurrent_state(
    state: RecurrentState | None,
    reset_mask: torch.Tensor,
) -> RecurrentState | None:
    """Zero selected batch rows without modifying the caller-owned state.

    ``reset_mask`` identifies samples that start a new sequence. Non-reset
    rows retain both their values and their autograd connection, while reset
    rows are replaced functionally so that gradients cannot cross a sequence
    boundary. Passing ``None`` is valid before the recurrent state has been
    initialized.
    """

    if not isinstance(reset_mask, torch.Tensor):
        raise TypeError("reset_mask must be a tensor")
    if reset_mask.ndim != 1:
        raise ValueError(
            f"reset_mask must have shape [B], got {tuple(reset_mask.shape)}"
        )
    if reset_mask.dtype != torch.bool:
        raise ValueError(f"reset_mask must have dtype bool, got {reset_mask.dtype}")
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        _validate_state_reset_tensor(state, reset_mask, "recurrent state")
        return _zero_reset_state_tensor(state, reset_mask)
    if isinstance(state, tuple) and len(state) == 2 and all(
        isinstance(value, torch.Tensor) for value in state
    ):
        hidden, cell = state
        _validate_state_reset_tensor(hidden, reset_mask, "ConvLSTM hidden state")
        _validate_state_reset_tensor(cell, reset_mask, "ConvLSTM cell state")
        if hidden.shape != cell.shape:
            raise ValueError("ConvLSTM hidden and cell states must have the same shape")
        if hidden.device != cell.device:
            raise ValueError("ConvLSTM hidden and cell states must share a device")
        if hidden.dtype != cell.dtype:
            raise ValueError("ConvLSTM hidden and cell states must have the same dtype")
        return (
            _zero_reset_state_tensor(hidden, reset_mask),
            _zero_reset_state_tensor(cell, reset_mask),
        )
    raise TypeError("recurrent state must be a tensor or an (hidden, cell) tensor pair")


class ConvGRUCell(nn.Module):
    """A convolutional GRU cell that preserves the patch-grid layout."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("recurrent channel counts must be positive")
        _validate_kernel_size(kernel_size)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        combined_channels = input_channels + hidden_channels
        self.gates = nn.Conv2d(
            combined_channels,
            hidden_channels * 2,
            kernel_size,
            padding=padding,
            bias=bias,
        )
        self.candidate = nn.Conv2d(
            combined_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
            bias=bias,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.gates.weight)
        nn.init.xavier_uniform_(self.candidate.weight)
        if self.gates.bias is not None:
            nn.init.zeros_(self.gates.bias)
        if self.candidate.bias is not None:
            nn.init.zeros_(self.candidate.bias)

    def initial_state(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"ConvGRU input must have shape [B,{self.input_channels},H,W]"
            )
        return x.new_zeros(
            x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1]
        )

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"ConvGRU input must have shape [B,{self.input_channels},H,W]"
            )
        hidden = self.initial_state(x) if state is None else state
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("ConvGRU state must be a tensor")
        _validate_state_tensor(hidden, x, self.hidden_channels, "ConvGRU state")

        reset, update = self.gates(torch.cat((x, hidden), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = self.candidate(torch.cat((x, reset * hidden), dim=1)).tanh()
        return update * hidden + (1.0 - update) * candidate


class ConvLSTMCell(nn.Module):
    """A convolutional LSTM cell that preserves the patch-grid layout."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("recurrent channel counts must be positive")
        _validate_kernel_size(kernel_size)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size,
            padding=padding,
            bias=bias,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.gates.weight)
        if self.gates.bias is not None:
            nn.init.zeros_(self.gates.bias)
            with torch.no_grad():
                self.gates.bias[
                    self.hidden_channels : self.hidden_channels * 2
                ].fill_(1.0)

    def initial_state(self, x: torch.Tensor) -> ConvLSTMState:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"ConvLSTM input must have shape [B,{self.input_channels},H,W]"
            )
        shape = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        return x.new_zeros(shape), x.new_zeros(shape)

    def forward(
        self, x: torch.Tensor, state: ConvLSTMState | None = None
    ) -> ConvLSTMState:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"ConvLSTM input must have shape [B,{self.input_channels},H,W]"
            )
        if state is None:
            hidden, cell = self.initial_state(x)
        elif isinstance(state, tuple) and len(state) == 2:
            hidden, cell = state
        else:
            raise TypeError("ConvLSTM state must be an (hidden, cell) tensor pair")
        if not isinstance(hidden, torch.Tensor) or not isinstance(cell, torch.Tensor):
            raise TypeError("ConvLSTM hidden and cell states must be tensors")
        _validate_state_tensor(hidden, x, self.hidden_channels, "ConvLSTM hidden state")
        _validate_state_tensor(cell, x, self.hidden_channels, "ConvLSTM cell state")

        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat((x, hidden), dim=1)
        ).chunk(4, dim=1)
        input_gate = input_gate.sigmoid()
        forget_gate = forget_gate.sigmoid()
        output_gate = output_gate.sigmoid()
        candidate = candidate.tanh()
        next_cell = forget_gate * cell + input_gate * candidate
        next_hidden = output_gate * next_cell.tanh()
        return next_hidden, next_cell


class RecurrentVJEPA21EventVisionTransformer(VJEPA21EventVisionTransformer):
    """V-JEPA 2.1 event encoder with causal recurrence on its patch grid.

    ``pre_encoder`` preserves the original R0 path: current masked patch
    embeddings are zeroed, spatial recurrence updates the complete patch grid,
    and retained recurrent tokens then enter the transformer. ``post_encoder``
    instead encodes a full frame with the ViT before updating recurrent state;
    an optional keep mask is applied only after recurrence. The latter is the
    causal future-prediction path, where masking must not alter temporal memory.

    State is always supplied and returned by the caller; the module never
    retains sequence state in parameters or buffers. ``forward_frame`` and
    ``forward_frame_intermediates`` explicitly bypass recurrence, including for
    an EMA copy of this module. ``forward_frame_and_recurrent`` exposes both the
    frame and recurrent tokens from one frame-encoder execution in
    ``post_encoder`` mode.

    The inherited ``forward`` and ``forward_intermediates`` APIs remain
    stateless and start from a zero state. Sequence training should use
    ``forward_recurrent`` or ``forward_recurrent_intermediates`` instead.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        input_channels: int = 10,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        scale_dim: int = 128,
        supervision_layers: tuple[int, ...] = (),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        recurrent_cell: RecurrentCellKind = "conv_lstm",
        recurrent_kernel_size: int = 3,
        recurrent_placement: RecurrentPlacement = "pre_encoder",
    ) -> None:
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            input_channels=input_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            scale_dim=scale_dim,
            supervision_layers=supervision_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        normalized_cell = recurrent_cell.replace("_", "")
        if normalized_cell == "convlstm":
            cell: ConvLSTMCell | ConvGRUCell = ConvLSTMCell(
                embed_dim, embed_dim, recurrent_kernel_size
            )
        elif normalized_cell == "convgru":
            cell = ConvGRUCell(embed_dim, embed_dim, recurrent_kernel_size)
        else:
            raise ValueError("recurrent_cell must select ConvLSTM or ConvGRU")
        if recurrent_placement not in {"pre_encoder", "post_encoder"}:
            raise ValueError(
                "recurrent_placement must be pre_encoder or post_encoder"
            )
        self.recurrent_cell_type = normalized_cell
        self.recurrent_placement = recurrent_placement
        self.recurrent_cell = cell

    @staticmethod
    def detach_state(state: RecurrentState | None) -> RecurrentState | None:
        return detach_recurrent_state(state)

    def _update_state(
        self, patches: torch.Tensor, state: RecurrentState | None
    ) -> tuple[torch.Tensor, RecurrentState]:
        if isinstance(self.recurrent_cell, ConvLSTMCell):
            if state is not None and not isinstance(state, tuple):
                raise TypeError("ConvLSTM encoder state must be an (hidden, cell) pair")
            next_state = self.recurrent_cell(patches, state)
            return next_state[0], next_state
        if state is not None and not isinstance(state, torch.Tensor):
            raise TypeError("ConvGRU encoder state must be a tensor")
        next_hidden = self.recurrent_cell(patches, state)
        return next_hidden, next_hidden

    def _forward_frame_intermediates(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None,
        *,
        require_configured_size: bool,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, int]]:
        """Run only the frame ViT, bypassing recurrent dynamic dispatch."""

        return VJEPA21EventVisionTransformer._forward_intermediates(
            self,
            x,
            scale_embedding,
            context_keep_mask,
            require_configured_size=require_configured_size,
        )

    def forward_frame_intermediates(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Return frame-ViT features without executing the recurrent cell."""

        outputs, _ = self._forward_frame_intermediates(
            x,
            scale_embedding,
            context_keep_mask,
            require_configured_size=True,
        )
        return outputs

    def forward_frame(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the final frame-ViT tokens with recurrence fully bypassed."""

        return self.forward_frame_intermediates(
            x, scale_embedding, context_keep_mask
        )[-1]

    def forward_frame_feature_map(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return a recurrence-free frame feature map at a dynamic resolution."""

        outputs, grid_size = self._forward_frame_intermediates(
            x,
            scale_embedding,
            None,
            require_configured_size=False,
        )
        return outputs[-1].transpose(1, 2).reshape(
            x.shape[0], self.embed_dim, grid_size[0], grid_size[1]
        )

    @staticmethod
    def _returned_state(
        state: RecurrentState,
        *,
        detach_state: bool,
    ) -> RecurrentState:
        returned = detach_recurrent_state(state) if detach_state else state
        if returned is None:
            raise RuntimeError("recurrent cell did not produce a state")
        return returned

    def _forward_pre_encoder_recurrent_intermediates(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None,
        state: RecurrentState | None,
        *,
        detach_state: bool,
        require_configured_size: bool,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, int], RecurrentState]:
        if x.ndim != 4:
            raise ValueError("expected event images with shape [B, C, H, W]")
        if require_configured_size and tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(
                f"expected images [B,C,{self.image_size[0]},{self.image_size[1]}]"
            )
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError("input image dimensions must be divisible by patch_size")

        batch_size = x.shape[0]
        patches_2d = self.patch_embed(x)
        grid_size = (patches_2d.shape[-2], patches_2d.shape[-1])
        if context_keep_mask is not None:
            validate_patch_mask(
                context_keep_mask, batch_size, grid_size[0] * grid_size[1]
            )
            if context_keep_mask.device != patches_2d.device:
                raise ValueError("context mask and patch embeddings must share a device")
            keep_grid = context_keep_mask.reshape(
                batch_size, 1, grid_size[0], grid_size[1]
            )
            patches_2d = patches_2d.masked_fill(~keep_grid, 0.0)

        # Update the complete state grid before gathering retained tokens. The
        # current masked embeddings are already zero, while history remains in
        # the externally supplied full-grid state.
        recurrent_grid, next_state = self._update_state(patches_2d, state)
        patches = recurrent_grid.flatten(2).transpose(1, 2)
        positions = self._patch_positions(batch_size, grid_size, x.device)
        if context_keep_mask is not None:
            patches = gather_equal_count(patches, context_keep_mask)
            positions = gather_equal_count(positions, context_keep_mask)

        scale_token = self.scale_projection(scale_embedding).unsqueeze(1)
        sequence = torch.cat((scale_token, patches), dim=1)
        outputs: list[torch.Tensor] = []
        selected = set(self.supervision_layers)
        for index, block in enumerate(self.blocks):
            sequence = block(sequence, positions)
            if index in selected:
                outputs.append(self.norm(sequence)[:, 1:, :])

        returned_state = self._returned_state(
            next_state, detach_state=detach_state
        )
        return tuple(outputs), grid_size, returned_state

    def _forward_post_encoder_recurrent_intermediates(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None,
        state: RecurrentState | None,
        *,
        detach_state: bool,
        require_configured_size: bool,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[int, int],
        RecurrentState,
        torch.Tensor,
    ]:
        # The frame encoder deliberately sees the full event window. A keep mask
        # may select returned tokens, but it never erases information before the
        # recurrent update or changes the full-grid temporal state.
        frame_outputs, grid_size = self._forward_frame_intermediates(
            x,
            scale_embedding,
            None,
            require_configured_size=require_configured_size,
        )
        frame_tokens = frame_outputs[-1]
        batch_size = x.shape[0]
        frame_grid = frame_tokens.transpose(1, 2).reshape(
            batch_size, self.embed_dim, grid_size[0], grid_size[1]
        )
        recurrent_grid, next_state = self._update_state(frame_grid, state)
        recurrent_tokens = recurrent_grid.flatten(2).transpose(1, 2)

        returned_frame_tokens = frame_tokens
        if context_keep_mask is not None:
            validate_patch_mask(
                context_keep_mask, batch_size, grid_size[0] * grid_size[1]
            )
            if context_keep_mask.device != recurrent_tokens.device:
                raise ValueError("context mask and recurrent tokens must share a device")
            returned_frame_tokens = gather_equal_count(
                returned_frame_tokens, context_keep_mask
            )
            recurrent_tokens = gather_equal_count(
                recurrent_tokens, context_keep_mask
            )

        returned_state = self._returned_state(
            next_state, detach_state=detach_state
        )
        return (recurrent_tokens,), grid_size, returned_state, returned_frame_tokens

    def _forward_recurrent_intermediates(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None,
        state: RecurrentState | None,
        *,
        detach_state: bool,
        require_configured_size: bool,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, int], RecurrentState]:
        if self.recurrent_placement == "pre_encoder":
            return self._forward_pre_encoder_recurrent_intermediates(
                x,
                scale_embedding,
                context_keep_mask,
                state,
                detach_state=detach_state,
                require_configured_size=require_configured_size,
            )
        outputs, grid_size, next_state, _ = (
            self._forward_post_encoder_recurrent_intermediates(
                x,
                scale_embedding,
                context_keep_mask,
                state,
                detach_state=detach_state,
                require_configured_size=require_configured_size,
            )
        )
        return outputs, grid_size, next_state

    def _forward_intermediates(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None,
        *,
        require_configured_size: bool,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, int]]:
        outputs, grid_size, _ = self._forward_recurrent_intermediates(
            x,
            scale_embedding,
            context_keep_mask,
            None,
            detach_state=False,
            require_configured_size=require_configured_size,
        )
        return outputs, grid_size

    def forward_recurrent_intermediates(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None = None,
        *,
        state: RecurrentState | None = None,
        detach_state: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], RecurrentState]:
        outputs, _, next_state = self._forward_recurrent_intermediates(
            x,
            scale_embedding,
            context_keep_mask,
            state,
            detach_state=detach_state,
            require_configured_size=True,
        )
        return outputs, next_state

    def forward_recurrent(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None = None,
        *,
        state: RecurrentState | None = None,
        detach_state: bool = False,
    ) -> tuple[torch.Tensor, RecurrentState]:
        outputs, next_state = self.forward_recurrent_intermediates(
            x,
            scale_embedding,
            context_keep_mask,
            state=state,
            detach_state=detach_state,
        )
        return outputs[-1], next_state

    def forward_frame_and_recurrent(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None = None,
        *,
        state: RecurrentState | None = None,
        detach_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, RecurrentState]:
        """Return frame tokens, recurrent tokens, and state from one ViT pass.

        This fused API is intentionally limited to ``post_encoder`` placement.
        In ``pre_encoder`` placement the transformer consumes recurrent patches,
        so a recurrence-free frame latent cannot share the same transformer pass.
        """

        if self.recurrent_placement != "post_encoder":
            raise ValueError(
                "forward_frame_and_recurrent requires post_encoder placement"
            )
        outputs, _, next_state, frame_tokens = (
            self._forward_post_encoder_recurrent_intermediates(
                x,
                scale_embedding,
                context_keep_mask,
                state,
                detach_state=detach_state,
                require_configured_size=True,
            )
        )
        return frame_tokens, outputs[-1], next_state

    def forward_feature_map_recurrent(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        *,
        state: RecurrentState | None = None,
        detach_state: bool = False,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Return a recurrent token grid at a detection-time padded resolution."""

        outputs, grid_size, next_state = self._forward_recurrent_intermediates(
            x,
            scale_embedding,
            None,
            state,
            detach_state=detach_state,
            require_configured_size=False,
        )
        tokens = outputs[-1]
        feature_map = tokens.transpose(1, 2).reshape(
            x.shape[0], self.embed_dim, grid_size[0], grid_size[1]
        )
        return feature_map, next_state


__all__ = [
    "ConvGRUCell",
    "ConvLSTMCell",
    "ConvLSTMState",
    "RecurrentCellKind",
    "RecurrentPlacement",
    "RecurrentState",
    "RecurrentVJEPA21EventVisionTransformer",
    "detach_recurrent_state",
    "reset_recurrent_state",
]
