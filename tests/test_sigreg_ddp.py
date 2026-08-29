from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import DistributedDataParallel

from event_window_jepa.losses.sigreg import ProjectedSIGReg


def _module() -> ProjectedSIGReg:
    return ProjectedSIGReg(
        input_dim=4,
        projection_dim=3,
        hidden_dim=5,
        num_slices=7,
        num_frequencies=5,
        seed=43,
    )


def _global_features() -> torch.Tensor:
    generator = torch.Generator().manual_seed(47)
    features = torch.randn(4, 4, generator=generator)
    features[2:] = float("nan")
    return features


def _global_feature_chunks() -> torch.Tensor:
    generator = torch.Generator().manual_seed(53)
    chunks = torch.randn(2, 4, 4, generator=generator)
    chunks[:, 2:] = float("nan")
    return chunks


def _ddp_worker(
    rank: int,
    world_size: int,
    initialization_file: str,
    output_directory: str,
) -> None:
    distributed.init_process_group(
        "gloo",
        init_method=f"file://{initialization_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        module = DistributedDataParallel(_module())
        features = _global_features()[rank * 2 : (rank + 1) * 2].clone()
        features.requires_grad_(True)
        valid = torch.tensor([True, True]) if rank == 0 else torch.tensor([False, False])
        output = module(features, valid)
        output.loss.backward()
        first_weight = module.module.projector.network[0].weight
        torch.save(
            {
                "loss": output.loss.detach(),
                "effective_samples": output.effective_samples,
                "parameter_gradient": first_weight.grad.detach(),
                "input_gradient": features.grad.detach(),
            },
            Path(output_directory) / f"rank-{rank}.pt",
        )
    finally:
        distributed.destroy_process_group()


def _ddp_no_sync_worker(
    rank: int,
    world_size: int,
    initialization_file: str,
    output_directory: str,
) -> None:
    distributed.init_process_group(
        "gloo",
        init_method=f"file://{initialization_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        module = DistributedDataParallel(_module())
        chunks = _global_feature_chunks()
        valid = (
            torch.tensor([True, True])
            if rank == 0
            else torch.tensor([False, False])
        )
        for chunk_index in range(chunks.shape[0]):
            features = chunks[
                chunk_index, rank * 2 : (rank + 1) * 2
            ].clone()
            features.requires_grad_(True)
            synchronization = (
                module.no_sync()
                if chunk_index + 1 < chunks.shape[0]
                else nullcontext()
            )
            with synchronization:
                (module(features, valid).loss / chunks.shape[0]).backward()
        first_weight = module.module.projector.network[0].weight
        torch.save(
            first_weight.grad.detach(),
            Path(output_directory) / f"no-sync-rank-{rank}.pt",
        )
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(
    not distributed.is_available() or not distributed.is_gloo_available(),
    reason="CPU gloo is required for the SIGReg DDP contract",
)
def test_sigreg_ddp_matches_global_batch_with_one_zero_support_rank(tmp_path) -> None:
    world_size = 2
    initialization_file = tmp_path / "gloo-init"
    multiprocessing.spawn(
        _ddp_worker,
        args=(world_size, str(initialization_file), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    reference = _module()
    features = _global_features().requires_grad_(True)
    valid = torch.tensor([True, True, False, False])
    reference_output = reference(features, valid)
    reference_output.loss.backward()
    reference_parameter_gradient = reference.projector.network[0].weight.grad

    for rank in range(world_size):
        result = torch.load(
            tmp_path / f"rank-{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        torch.testing.assert_close(result["loss"], reference_output.loss)
        assert result["effective_samples"].item() == 2
        torch.testing.assert_close(
            result["parameter_gradient"], reference_parameter_gradient
        )
        expected_input_gradient = (
            features.grad[rank * 2 : (rank + 1) * 2] * world_size
        )
        torch.testing.assert_close(
            result["input_gradient"], expected_input_gradient
        )


@pytest.mark.skipif(
    not distributed.is_available() or not distributed.is_gloo_available(),
    reason="CPU gloo is required for the SIGReg DDP contract",
)
def test_sigreg_ddp_no_sync_accumulates_all_tbptt_chunks(tmp_path) -> None:
    world_size = 2
    initialization_file = tmp_path / "gloo-no-sync-init"
    multiprocessing.spawn(
        _ddp_no_sync_worker,
        args=(world_size, str(initialization_file), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    reference = _module()
    chunks = _global_feature_chunks()
    valid = torch.tensor([True, True, False, False])
    for chunk in chunks:
        (reference(chunk, valid).loss / chunks.shape[0]).backward()
    reference_gradient = reference.projector.network[0].weight.grad

    for rank in range(world_size):
        gradient = torch.load(
            tmp_path / f"no-sync-rank-{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        torch.testing.assert_close(gradient, reference_gradient)
