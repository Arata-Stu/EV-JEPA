from __future__ import annotations

from typing import Literal

import pytest
import torch

from event_window_jepa.models.ema_encoder import make_ema_copy
from event_window_jepa.models.recurrent_vjepa21_event_vit import (
    ConvGRUCell,
    ConvLSTMCell,
    RecurrentVJEPA21EventVisionTransformer,
    detach_recurrent_state,
    reset_recurrent_state,
)


def _encoder(
    recurrent_cell: Literal["convlstm", "convgru"] = "convlstm",
) -> RecurrentVJEPA21EventVisionTransformer:
    return RecurrentVJEPA21EventVisionTransformer(
        image_size=(16, 16),
        patch_size=8,
        input_channels=10,
        embed_dim=32,
        depth=2,
        num_heads=4,
        scale_dim=16,
        supervision_layers=(0, 1),
        recurrent_cell=recurrent_cell,
    )


def test_conv_cells_create_input_matched_states_and_validate_shapes() -> None:
    x = torch.randn(2, 8, 3, 4, dtype=torch.float64)

    gru = ConvGRUCell(8, 12).double()
    gru_state = gru(x)
    assert gru_state.shape == (2, 12, 3, 4)
    assert gru_state.dtype == x.dtype
    assert gru_state.device == x.device
    with pytest.raises(ValueError, match="shape"):
        gru(x, torch.zeros(2, 12, 3, 3, dtype=x.dtype))

    lstm = ConvLSTMCell(8, 12).double()
    hidden, cell = lstm(x)
    assert hidden.shape == cell.shape == (2, 12, 3, 4)
    assert hidden.dtype == cell.dtype == x.dtype
    assert hidden.device == cell.device == x.device
    with pytest.raises(ValueError, match="dtype"):
        lstm(x, (hidden.float(), cell))


@pytest.mark.parametrize("cell_kind", ["convlstm", "convgru"])
def test_recurrent_encoder_zeroes_current_masked_patches_before_spatial_update(
    cell_kind: Literal["convlstm", "convgru"],
) -> None:
    encoder = _encoder(cell_kind)
    x = torch.randn(1, 10, 16, 16)
    changed_only_in_masked_patches = x.clone()
    changed_only_in_masked_patches[:, :, :8, 8:] += 1000.0
    changed_only_in_masked_patches[:, :, 8:, :8] -= 1000.0
    scale = torch.randn(1, 16)
    keep_mask = torch.tensor([[1, 0, 0, 1]], dtype=torch.bool)

    original_layers, original_state = encoder.forward_recurrent_intermediates(
        x, scale, keep_mask
    )
    changed_layers, changed_state = encoder.forward_recurrent_intermediates(
        changed_only_in_masked_patches, scale, keep_mask
    )

    assert all(layer.shape == (1, 2, 32) for layer in original_layers)
    assert all(
        torch.equal(a, b)
        for a, b in zip(original_layers, changed_layers, strict=True)
    )
    if isinstance(original_state, tuple):
        assert isinstance(changed_state, tuple)
        assert original_state[0].shape == original_state[1].shape == (1, 32, 2, 2)
        assert torch.equal(original_state[0], changed_state[0])
        assert torch.equal(original_state[1], changed_state[1])
    else:
        assert isinstance(changed_state, torch.Tensor)
        assert original_state.shape == (1, 32, 2, 2)
        assert torch.equal(original_state, changed_state)


def test_explicit_state_supports_bptt_and_detached_tbptt_boundaries() -> None:
    encoder = _encoder("convlstm")
    scale = torch.randn(1, 16)
    first_tokens, first_state = encoder.forward_recurrent(
        torch.randn(1, 10, 16, 16), scale
    )
    assert isinstance(first_state, tuple)
    assert first_state[0].grad_fn is not None
    assert first_state[1].grad_fn is not None

    second_tokens, detached_state = encoder.forward_recurrent(
        torch.randn(1, 10, 16, 16),
        scale,
        state=first_state,
        detach_state=True,
    )
    assert isinstance(detached_state, tuple)
    assert detached_state[0].grad_fn is None
    assert detached_state[1].grad_fn is None

    # Detaching the returned memory must not detach the current-step objective.
    (first_tokens.mean() + second_tokens.mean()).backward()
    assert encoder.recurrent_cell.gates.weight.grad is not None

    manually_detached = detach_recurrent_state(first_state)
    assert isinstance(manually_detached, tuple)
    assert all(value.grad_fn is None for value in manually_detached)


def test_reset_recurrent_state_preserves_unselected_rows_and_inputs() -> None:
    reset_mask = torch.tensor([False, True, False], dtype=torch.bool)
    hidden = torch.arange(24, dtype=torch.float64).reshape(3, 2, 2, 2)
    cell = hidden + 100.0
    hidden_before = hidden.clone()
    cell_before = cell.clone()

    reset_gru = reset_recurrent_state(hidden, reset_mask)
    reset_lstm = reset_recurrent_state((hidden, cell), reset_mask)

    assert isinstance(reset_gru, torch.Tensor)
    assert isinstance(reset_lstm, tuple)
    assert torch.equal(hidden, hidden_before)
    assert torch.equal(cell, cell_before)
    assert torch.equal(reset_gru[~reset_mask], hidden[~reset_mask])
    assert torch.count_nonzero(reset_gru[reset_mask]) == 0
    for reset_value, original in zip(reset_lstm, (hidden, cell), strict=True):
        assert reset_value.dtype == original.dtype
        assert reset_value.device == original.device
        assert torch.equal(reset_value[~reset_mask], original[~reset_mask])
        assert torch.count_nonzero(reset_value[reset_mask]) == 0


def test_reset_recurrent_state_validates_mask_and_state_metadata() -> None:
    state = torch.ones(2, 3, 4, 5)

    assert reset_recurrent_state(None, torch.zeros(2, dtype=torch.bool)) is None
    with pytest.raises(ValueError, match="shape"):
        reset_recurrent_state(state, torch.zeros(2, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="dtype bool"):
        reset_recurrent_state(state, torch.zeros(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="shape"):
        reset_recurrent_state(state, torch.zeros(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="same shape"):
        reset_recurrent_state(
            (state, torch.ones(2, 3, 4, 4)), torch.zeros(2, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="same dtype"):
        reset_recurrent_state(
            (state, state.double()), torch.zeros(2, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="reset_mask must be on"):
        reset_recurrent_state(
            state, torch.zeros(2, dtype=torch.bool, device="meta")
        )


def test_feedforward_api_is_stateless_and_state_is_not_checkpointed() -> None:
    encoder = _encoder("convgru")
    x = torch.randn(1, 10, 16, 16)
    scale = torch.randn(1, 16)
    state_dict_keys = tuple(encoder.state_dict())

    output = encoder(x, scale)
    recurrent_output, state = encoder.forward_recurrent(x, scale)

    assert isinstance(output, torch.Tensor)
    assert output.shape == recurrent_output.shape == (1, 4, 32)
    assert torch.equal(output, recurrent_output)
    assert isinstance(state, torch.Tensor)
    assert tuple(encoder.state_dict()) == state_dict_keys

    target = make_ema_copy(encoder)
    assert isinstance(target, RecurrentVJEPA21EventVisionTransformer)
    assert tuple(target.state_dict()) == state_dict_keys
    assert all(not parameter.requires_grad for parameter in target.parameters())
