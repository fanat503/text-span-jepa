# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Distributed-training helpers for `src/train.py`.

Why this module exists
----------------------
`scripts/wikitext/train_ddp.sh` runs `torchrun --nproc_per_node=N -m src.train`,
and before this module `torch.distributed` was imported *nowhere* in `src/`.
That made the script produce N independent replicas plus one unusable
checkpoint file, silently:

1. No process group, no all-reduce. Every rank optimised its own model. The
   "multi-GPU" run was N single-GPU runs writing to the same directory.
2. `src/train.py` resolved `cuda` with no `LOCAL_RANK`, so all ranks landed on
   `cuda:0`.
3. `logging.folder` was identical on every rank, so all ranks wrote
   `params-*.yaml`, `train_log.csv`, `best.pt` and `checkpoint-latest.pth.tar`
   concurrently and non-atomically. A `torch.save` interrupted halfway leaves a
   file that `safe_torch_load` refuses to read — which, after the
   `load_checkpoint` hardening in `src/train.py`, now stops the run instead of
   silently restarting it.
4. `seed_everything(42)` was unconditional and `make_dataloader` had no
   `sampler` argument, so a `DistributedSampler` could not even be injected.
5. Workspace subspaces split parameter/buffer: `jawp.workspace_Q` is a
   `Parameter` (all-reduced, so it converges to the mean) but
   `rdc.workspace_Q` and `wsd.target_Q` are **buffers** mutated in place. With
   DDP's default `broadcast_buffers=True`, DDP overwrites every rank's buffer
   with rank 0's at the top of each forward, so the mechanism reads a
   rank-0 value that is one step stale — never an average, never a crash.
   See `BUFFER_SYNC_DECISION` below.

Import-clean
------------
`torch.distributed` is imported lazily, inside the functions that need it. On a
build without distributed support the module still imports, every detection
helper still answers correctly, and the single-process path is unaffected. All
environment and rank logic is a pure function of an injectable mapping, so the
whole surface is unit-testable on a CPU-only box with no process group.

Not yet wired
-------------
Nothing in `src/train.py` calls this module yet; a follow-up wires it into
`main()`. Everything here is therefore additive and import-inert.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import time

import torch

__all__ = [
    "BUFFER_SYNC_DECISION",
    "FIND_UNUSED_PARAMETERS",
    "MECHANISM_BUFFERS_NEEDING_CONSENSUS",
    "all_reduce_buffers",
    "atomic_torch_save",
    "atomic_write_bytes",
    "atomic_write_text",
    "barrier",
    "cleanup",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "init_process_group",
    "is_distributed",
    "is_distributed_available",
    "is_launched_by_torchrun",
    "is_main_process",
    "log_dir_lock",
    "make_distributed_sampler",
    "mechanism_buffers_needing_consensus",
    "rank_aware_seed",
    "rank_local_device",
    "reduce_loss",
    "save_on_rank0",
    "unwrap_model",
]

RANK_ENV = "RANK"
LOCAL_RANK_ENV = "LOCAL_RANK"
WORLD_SIZE_ENV = "WORLD_SIZE"
MASTER_ADDR_ENV = "MASTER_ADDR"
MASTER_PORT_ENV = "MASTER_PORT"
LOCK_FILENAME = ".write.lock"


# ═══════════════════════════════════════════════════════════════════
#  Detection
# ═══════════════════════════════════════════════════════════════════


def _env(env=None):
    return os.environ if env is None else env


def is_launched_by_torchrun(env=None):
    """True when `torchrun` / `torch.distributed.launch` set the rank variables.

    `RANK` alone is enough; `torch.distributed.launch` sets `LOCAL_RANK` too but
    so does some unrelated tooling, and `RANK` is the one the launcher contract
    guarantees.
    """
    env = _env(env)
    return env.get(RANK_ENV) is not None and env.get(WORLD_SIZE_ENV) is not None


def is_distributed_available():
    """True when this torch build can create a process group. Never raises."""
    try:
        import torch.distributed as dist
    except Exception:  # pragma: no cover - build without distributed support
        return False
    return bool(dist.is_available())


def _dist_is_live():
    """True when a process group actually exists right now."""
    try:
        import torch.distributed as dist
    except Exception:  # pragma: no cover - build without distributed support
        return False
    return bool(dist.is_available() and dist.is_initialized())


def is_distributed(env=None):
    """True when a process group is live, or a launcher is asking for one.

    The second clause covers the window between "torchrun started us" and "we
    called `init_process_group`", so a sampler can be built before the group
    exists.
    """
    return _dist_is_live() or is_launched_by_torchrun(env)


def get_rank(env=None):
    """Global rank, 0 outside a distributed run."""
    if _dist_is_live():
        import torch.distributed as dist

        return int(dist.get_rank())
    raw = _env(env).get(RANK_ENV)
    return int(raw) if raw is not None else 0


def get_local_rank(env=None):
    """Rank within this node, 0 outside a distributed run.

    This is the index that picks the GPU. Using the *global* rank would send
    rank 4 of a 2-node job to `cuda:0` on its node and collide with rank 0.

    Only the launcher environment knows the local rank — the process group
    reports a global one — so `LOCAL_RANK` is read first and the global rank is
    the fallback.
    """
    raw = _env(env).get(LOCAL_RANK_ENV)
    if raw is not None:
        return int(raw)
    return get_rank(env)


def get_world_size(env=None):
    """Number of processes in the job, 1 outside a distributed run."""
    if _dist_is_live():
        import torch.distributed as dist

        return int(dist.get_world_size())
    raw = _env(env).get(WORLD_SIZE_ENV)
    return int(raw) if raw is not None else 1


def is_main_process(env=None):
    """True on rank 0. Every write and every log line must be behind this."""
    return get_rank(env) == 0


def rank_aware_seed(base_seed, env=None):
    """`base_seed + rank`.

    Every rank must initialise *differently*: identical seeds mean identical
    span masks and identical Gumbel draws, so the N ranks compute N copies of
    the same gradient and the all-reduce is a no-op that merely wastes the
    hardware. Disjoint streams are what make the averaged gradient a better
    estimate than any single replica's.
    """
    return int(base_seed) + get_rank(env)


# ═══════════════════════════════════════════════════════════════════
#  Device
# ═══════════════════════════════════════════════════════════════════


def rank_local_device(env=None, **kwargs):
    """This rank's device, honouring `LOCAL_RANK`.

    Delegates to `src.train.resolve_device`, which owns the documented priority
    chain (TPU > DDP GPU > CUDA > CPU) and the "PJRT_DEVICE set but
    pytorch_xla missing" error. Imported lazily so `src.utils` never depends on
    the training entry point at module-import time.
    """
    from src.train import resolve_device

    return resolve_device(env=_env(env), **kwargs)


# ═══════════════════════════════════════════════════════════════════
#  Process group lifecycle
# ═══════════════════════════════════════════════════════════════════


def init_process_group(backend=None, timeout_seconds=1800, env=None):
    """Create the default process group from the launcher's environment.

    Returns `(rank, world_size, backend_name)`. Safe to call when not launched
    by torchrun: it returns `(0, 1, None)` and does nothing, so callers do not
    need a branch.

    `MASTER_ADDR` / `MASTER_PORT` are required by the default `env://` init
    method. A missing one is an actionable error here rather than a hang in
    `init_process_group` itself.
    """
    env = _env(env)
    if not is_launched_by_torchrun(env):
        return 0, 1, None
    if not is_distributed_available():
        raise RuntimeError(
            "torchrun set RANK/WORLD_SIZE but this torch build has no "
            "torch.distributed support. Install a torch build with distributed "
            "support, or launch without torchrun.",
        )
    for key in (MASTER_ADDR_ENV, MASTER_PORT_ENV):
        if env.get(key) is None:
            raise RuntimeError(
                f"torchrun started this process but {key} is not set. The "
                f"launcher is expected to provide it; start the job with "
                f"`torchrun --nproc_per_node=N -m src.train ...` or set "
                f"{key} explicitly.",
            )
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
    return int(dist.get_rank()), int(dist.get_world_size()), str(backend)


def cleanup():
    """Tear the process group down if there is one. Never raises."""
    try:
        import torch.distributed as dist
    except Exception:  # pragma: no cover
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@contextlib.contextmanager
def process_group(backend=None, timeout_seconds=1800, env=None):
    """Context manager form of `init_process_group` + `cleanup`."""
    rank, world_size, name = init_process_group(
        backend=backend, timeout_seconds=timeout_seconds, env=env
    )
    try:
        yield rank, world_size, name
    finally:
        cleanup()


def barrier(env=None):
    """Synchronise every rank. No-op outside a distributed run."""
    if not is_distributed(env):
        return
    import torch.distributed as dist

    dist.barrier()


def reduce_loss(value, average=True, env=None):
    """All-reduce a scalar loss so rank 0 logs a global mean, not its own.

    Without this, the loss curve in `train_log.csv` is whichever rank happened
    to be writing it — i.e. an arbitrary single replica.
    """
    if not is_distributed(env):
        return value
    import torch.distributed as dist

    tensor = torch.tensor(float(value), dtype=torch.float64)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    result = tensor.item()
    return result / get_world_size(env) if average else result


def unwrap_model(model):
    """Strip a DDP wrapper, returning the underlying module."""
    return getattr(model, "module", model) if hasattr(model, "module") else model


# ═══════════════════════════════════════════════════════════════════
#  Buffer synchronisation — the parameter/buffer split
# ═══════════════════════════════════════════════════════════════════

# The workspace subspaces in this repo are split across the parameter/buffer
# line, and that split makes DDP's `broadcast_buffers` actively wrong:
#
#   jawp.workspace_Q   Parameter  -> all-reduced by DDP, converges to the mean
#   pcr.workspace_Q    Parameter  -> ditto
#   spc.freq_basis     Parameter  -> ditto
#   rdc.workspace_Q    buffer     -> NOT all-reduced
#   wsd.target_Q       buffer     -> NOT all-reduced, and is a LOSS INPUT
#
# With `broadcast_buffers=True` (the DDP default) every rank's copy of the
# buffer tensors is overwritten with rank 0's at the top of each forward. The
# mechanisms mutate those buffers in place *during* the forward and then read
# them back in the same forward, so what rank 1 actually uses is rank 0's
# value from the previous step: "rank 0's value, lagged one step", never an
# average, never a crash, no warning. For `wsd.target_Q` / `wsd.target_cov` the
# consequence is worse than a lag — the WSD gradient that gets all-reduced was
# computed against a reference subspace only rank 0 ever observed.
#
# DECISION: `broadcast_buffers=False`. It does not fabricate a consensus out of
# rank 0 either; each rank keeps the subspace it computed, and the all-reduced
# parameter gradients are an honest average over divergent-but-legitimate
# replicas. That is deterministic and matches single-process semantics modulo
# the replica split, which is the only defensible behaviour until the
# parameter/buffer split itself is resolved.
#
# The correct fix is structural, not a DDP flag: `rdc.workspace_Q` and
# `wsd.target_Q` should be Parameters (retracted like their JAWP/PCR/SPC
# siblings), or explicitly all-reduced. `all_reduce_buffers` below does the
# latter for callers who want consensus before that refactor lands.
BUFFER_SYNC_DECISION = "broadcast_buffers=False"

# Named because `src/models/*` is owned elsewhere: this is the list the
# mechanism refactor has to resolve, recorded here so the decision above can be
# checked against it.
MECHANISM_BUFFERS_NEEDING_CONSENSUS = (
    "rdc.workspace_Q",
    "wsd.target_Q",
    "wsd.target_cov",
)


def mechanism_buffers_needing_consensus(model):
    """Resolve `MECHANISM_BUFFERS_NEEDING_CONSENSUS` against a live model.

    Returns `(name, tensor)` pairs that are actually registered buffers, so a
    config that disables WSD or RDC contributes nothing instead of raising.
    """
    model = unwrap_model(model)
    buffers = dict(model.named_buffers())
    return [(n, buffers[n]) for n in MECHANISM_BUFFERS_NEEDING_CONSENSUS if n in buffers]


def all_reduce_buffers(model, average=True, env=None, all_reduce=None, world=None):
    """Average the mechanism buffers that hold per-rank workspace subspaces.

    `all_reduce` is injectable so this is testable without a process group. In a
    real run it defaults to `torch.distributed.all_reduce` on a float32 view;
    the caller's tensors are updated in place with `copy_` so buffer identity is
    preserved.
    """
    targets = mechanism_buffers_needing_consensus(model)
    if not targets:
        return []
    if all_reduce is None:
        if not _dist_is_live():
            raise RuntimeError(
                "all_reduce_buffers needs an initialised process group. Call it "
                "only when torch.distributed.is_initialized() is True.",
            )
        import torch.distributed as dist

        def all_reduce(tensor, op=None):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    world = get_world_size(env) if world is None else int(world)
    for _name, tensor in targets:
        flat = tensor.detach().reshape(-1).to(torch.float32)
        all_reduce(flat)
        if average and world > 1:
            flat = flat / world
        with torch.no_grad():
            tensor.copy_(flat.reshape(tensor.shape).to(tensor.dtype))
    return [name for name, _ in targets]


# ═══════════════════════════════════════════════════════════════════
#  find_unused_parameters
# ═══════════════════════════════════════════════════════════════════

# `True` is the only safe value here, and it is a constant only because there
# is no correct `False`.
#
# DDP's autograd hook walks the graph it recorded at the end of the forward and
# marks every parameter it did not see as "unused"; with the default `False`
# that is a hard error. This repo's mechanism graph is step-dependent by
# construction:
#
#   * `pcr.level_gates` and `spc.freq_basis` are only reached when their
#     branch fires (PCR's warmup / level selection, SPC's band adaptation), so
#     on any given step some cascades are not in the graph at all;
#   * `cmc` runs a SECOND `compute_loss` whose parameters are only reachable
#     through the bridge, and only on `cmc.should_compute(global_step)` steps;
#   * `src/train.py` runs a second, separate `backward()` for GAC after the
#     main one, on a graph the DDP reducer did not record;
#   * every mechanism is individually switchable from the config, so an
#     ablation legitimately removes parameters from the forward.
#
# A `False` would therefore crash intermittently, on a schedule that depends on
# the step counter — the worst possible failure mode for a long run. `True`
# costs one graph traversal per backward and cannot be wrong.
FIND_UNUSED_PARAMETERS = True


# ═══════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════


def make_distributed_sampler(dataset, shuffle=True, seed=0, drop_last=True, env=None):
    """A `DistributedSampler` when distributed, else `None` (keep RandomSampler).

    `num_replicas` / `rank` are taken from the live process group when one
    exists and from the launcher environment otherwise, so the sampler can be
    built before `init_process_group`.

    `seed` should be the *rank-aware* seed (`rank_aware_seed`): a shared seed
    across ranks makes every rank enumerate the same order, which is fine for
    the sampler (the shards are disjoint) but not for the models.
    """
    if not is_launched_by_torchrun(env):
        return None
    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(env),
        rank=get_rank(env),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


# ═══════════════════════════════════════════════════════════════════
#  Rank-0-only, atomic writes
# ═══════════════════════════════════════════════════════════════════


def atomic_write_bytes(path, payload):
    """Write via a same-directory temp file and `os.replace`.

    `os.replace` is atomic on POSIX and on Windows (MoveFileEx with
    REPLACE_EXISTING). A reader therefore sees either the old file or the
    complete new one, never a half-written `torch.save` — which is what made
    every checkpoint produced by the old DDP script unusable.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path, text, encoding="utf-8"):
    atomic_write_bytes(path, text.encode(encoding))


def atomic_torch_save(obj, path):
    """`torch.save` into a temp file, then rename into place."""
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_on_rank0(save_fn, *args, rank=None, env=None, **kwargs):
    """Call `save_fn(*args, **kwargs)` only on rank 0, atomically.

    `save_fn` must take the destination path as its **first positional
    argument**, or as a `path=` keyword — the convention of
    `src.train.save_checkpoint(path, model, ...)`. Note that `torch.save` is the
    other way round (`torch.save(obj, path)`), so it cannot be passed directly;
    wrap it.

    The temp file is created in the destination directory (same filesystem, so
    `os.replace` is atomic) and renamed into place, so a reader never observes
    a partial `torch.save`.

    Returns the final path on rank 0 and `None` everywhere else, so callers can
    assert without knowing the rank.
    """
    rank = get_rank(env) if rank is None else int(rank)
    if rank != 0:
        return None
    if args:
        path = args[0]
        tmp = f"{path}.tmp.rank0.{os.getpid()}"
        save_fn(tmp, *args[1:], **kwargs)
    else:
        path = kwargs["path"]
        tmp = f"{path}.tmp.rank0.{os.getpid()}"
        kwargs["path"] = tmp
        save_fn(**kwargs)
    os.replace(tmp, path)
    return path


# ═══════════════════════════════════════════════════════════════════
#  Log-directory lock
# ═══════════════════════════════════════════════════════════════════


def _lock_is_stale(lock_path, stale_after):
    try:
        age = time.time() - os.stat(lock_path).st_mtime
    except OSError:
        return False
    return age > stale_after


@contextlib.contextmanager
def log_dir_lock(log_dir, timeout=600.0, poll=0.5, owner="", env=None, stale_after=None):
    """Exclusive advisory lock over a log directory.

    Two different hazards, one primitive:

    * Ranks in one job: handled by `is_main_process` gating. The lock is not
      needed and is not taken for them.
    * Two *independent jobs* launched at the same `logging.folder`: nothing in
      torch stops them, and they would interleave `params-*.yaml`,
      `train_log.csv` and `checkpoint-latest.pth.tar`. This blocks the second
      one.

    Implemented with `O_CREAT | O_EXCL`, which is atomic on every filesystem
    this repo targets.

    Two independent timeouts, deliberately:

    * `timeout` — how long to wait for a *live* holder before giving up.
    * `stale_after` — how old an untouched lock file must be before we assume
      the holder was killed mid-write and break it. It must be much larger than
      the longest expected single write (a multi-gigabyte `torch.save` takes
      minutes), hence the 1-hour floor. Tying the two together is a bug: the
      wait would always end by declaring the holder dead, and the lock would
      never actually exclude anyone.

    Yields True if the lock was acquired by this caller. Does nothing outside a
    distributed run, so the single-process path is unchanged.
    """
    if not is_distributed(env):
        yield False
        return
    if stale_after is None:
        stale_after = max(3600.0, 10.0 * timeout)
    os.makedirs(log_dir, exist_ok=True)
    lock_path = os.path.join(log_dir, LOCK_FILENAME)
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_is_stale(lock_path, stale_after):
                with contextlib.suppress(OSError):
                    os.remove(lock_path)
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Could not acquire {lock_path!r} within {timeout}s. Another "
                    f"job is writing to {log_dir!r}; point logging.folder "
                    f"somewhere else or wait for it to finish.",
                )
            time.sleep(poll)
    try:
        os.write(fd, f"pid={os.getpid()} rank={get_rank(env)} owner={owner}\n".encode())
        os.close(fd)
        fd = None
        yield True
    finally:
        if fd is not None:  # pragma: no cover - only on a write failure
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            os.remove(lock_path)
