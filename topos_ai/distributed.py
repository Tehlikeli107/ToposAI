import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

HAS_FSDP = False
try:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

    HAS_FSDP = True
except ImportError:
    dist = None
    FSDP = None
    CPUOffload = None


def setup_distributed_topos(model: nn.Module, rank: int, world_size: int):
    """
    Wrap a model with PyTorch FSDP when distributed training is initialized.

    This is scaffolding for sharded training. It does not by itself validate
    multi-node throughput or trillion-parameter scaling.
    """
    del world_size

    if not HAS_FSDP:
        logger.warning("PyTorch FSDP is unavailable; returning the original model.")
        return model

    if not dist.is_initialized():
        logger.warning("torch.distributed is not initialized; returning the original model.")
        return model

    logger.info("Wrapping Topos model on rank %s with FSDP.", rank)
    return FSDP(model, cpu_offload=CPUOffload(offload_params=True))


def setup_expert_parallelism(model: nn.Module, rank: int, world_size: int):
    """
    Record expert-parallel universe ownership for MoE-style prototypes.

    Actual memory and throughput benefits require model code that consumes
    `local_universe_indices` plus distributed execution benchmarks.
    """
    if world_size <= 1:
        return model

    for _name, module in model.named_modules():
        if module.__class__.__name__ != "MultiUniverseToposAttention":
            continue

        num_universes = getattr(module, "num_universes", 0)
        if num_universes == 0:
            continue

        universes_per_rank = num_universes // world_size
        if universes_per_rank == 0:
            logger.warning(
                "Rank %s: world_size (%s) exceeds num_universes (%s); skipping expert shard assignment.",
                rank,
                world_size,
                num_universes,
            )
            continue

        start_idx = rank * universes_per_rank
        end_idx = start_idx + universes_per_rank if rank < world_size - 1 else num_universes

        module.local_universe_indices = list(range(start_idx, end_idx))
        logger.info("Rank %s owns universe indices: %s", rank, module.local_universe_indices)

    return model
