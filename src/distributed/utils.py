"""Distributed utility helpers: rank, world size, NCCL init, and simulation stubs."""

import os
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DISTRIBUTED_INITIALIZED = False


def init_distributed(backend: str = "nccl", rank: Optional[int] = None,
                     world_size: Optional[int] = None) -> bool:
    """Initialize torch.distributed if not already done.

    On CPU or when CUDA is unavailable, silently skips NCCL init.

    Returns:
        True if distributed was successfully initialized, False otherwise.
    """
    global _DISTRIBUTED_INITIALIZED
    if _DISTRIBUTED_INITIALIZED:
        return True

    if not torch.cuda.is_available():
        logger.info("CUDA unavailable — skipping torch.distributed init (CPU mode).")
        return False

    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size <= 1:
        logger.info("world_size=1, skipping distributed init.")
        return False

    try:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )
        _DISTRIBUTED_INITIALIZED = True
        logger.info(f"torch.distributed initialized: rank={rank}/{world_size}, backend={backend}")
        return True
    except Exception as e:
        logger.warning(f"torch.distributed init failed: {e}. Falling back to single-process.")
        return False


def is_distributed() -> bool:
    """Return True if torch.distributed is initialized and world_size > 1."""
    try:
        return (torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1)
    except Exception:
        return False


def get_rank() -> int:
    """Return the current process rank (0 if not distributed)."""
    if is_distributed():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Return total number of distributed processes (1 if not distributed)."""
    if is_distributed():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def barrier() -> None:
    """Synchronize all processes (no-op if not distributed)."""
    if is_distributed():
        torch.distributed.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """In-place all-reduce (sum) across all ranks. Returns the tensor."""
    if is_distributed():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from `src` rank to all other ranks."""
    if is_distributed():
        torch.distributed.broadcast(tensor, src=src)
    return tensor


def send_tensor(tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
    """Non-blocking send of a tensor to another rank."""
    if is_distributed():
        torch.distributed.isend(tensor, dst=dst, tag=tag).wait()


def recv_tensor(tensor: torch.Tensor, src: int, tag: int = 0) -> torch.Tensor:
    """Blocking receive of a tensor from another rank into pre-allocated buffer."""
    if is_distributed():
        torch.distributed.irecv(tensor, src=src, tag=tag).wait()
    return tensor


def compute_comm_bytes(tensor: torch.Tensor) -> int:
    """Return the number of bytes that would be sent for this tensor."""
    return tensor.nelement() * tensor.element_size()
