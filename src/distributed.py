"""Multi-GPU helpers (PyTorch DistributedDataParallel, DDP).

With DDP, one process runs per GPU. Each process trains on its own share of
the data; after every backward pass the gradients are averaged across the
processes, so all model copies stay identical. Started with torchrun:

    torchrun --standalone --nproc_per_node=2 -m src.train --config ... --out ...

Without torchrun (plain `python -m src.train`) everything runs as a single
process on one GPU or the CPU, exactly as before.

Rules used in src/train.py:
  - only the MAIN process (rank 0) writes files, logs and checkpoints;
  - decisions based on the clock (validate now? save? stop?) are made by the
    main process and broadcast, so all processes always agree; otherwise
    one could wait forever for the other.
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


class Distributed:
    def __init__(self):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.enabled = self.world_size > 1
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}" if cuda else "cpu")
        if self.enabled:
            # NCCL is the fast GPU backend; gloo works on CPU (local tests).
            # Long timeout: the other processes wait while rank 0 validates.
            dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(minutes=60))

    @property
    def is_main(self):
        return self.rank == 0

    def broadcast_flags(self, flags):
        """Make every process use the main process's list of True/False decisions."""
        if not self.enabled:
            return flags
        t = torch.tensor([int(f) for f in flags], device=self.device)
        dist.broadcast(t, src=0)
        return [bool(v) for v in t.tolist()]

    def mean(self, value):
        """Average a number over all processes (e.g. the training loss for the log)."""
        if not self.enabled:
            return value
        t = torch.tensor([float(value)], device=self.device)
        dist.all_reduce(t)
        return t.item() / self.world_size

    def barrier(self):
        """Wait until every process reaches this point."""
        if self.enabled:
            dist.barrier()

    def close(self):
        if self.enabled:
            dist.destroy_process_group()
