# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Gates for `src/utils/distributed.py`. No GPU, no process group, no launcher.

`scripts/wikitext/train_ddp.sh` runs `torchrun --nproc_per_node=N` while
`torch.distributed` was imported nowhere in `src/`. These tests pin the
decisions that module encodes — above all the two that silently corrupt a
multi-GPU run: `broadcast_buffers` (rank 0's value, lagged one step, never an
average) and concurrent non-atomic writes to a shared `logging.folder`.

Every rank-detection function takes an injected environment mapping, so the
whole surface is exercised by monkeypatching `os.environ` lookups rather than
by spawning processes. That is the point of the design: the failure this module
fixes was invisible precisely because it could not be observed from one process.
"""

import os

import pytest
import torch
import torch.nn as nn

from src.utils import distributed as D
from src.utils.seed import seed_everything


def _torchrun_env(rank, world=4, **extra):
    env = {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world)}
    env.update(extra)
    return env


def _single_env(**extra):
    env = {}
    env.update(extra)
    return env


class TestRankDetection:
    def test_single_process_is_rank_zero_of_one(self):
        env = _single_env()
        assert D.is_launched_by_torchrun(env) is False
        assert D.get_rank(env) == 0
        assert D.get_local_rank(env) == 0
        assert D.get_world_size(env) == 1
        assert D.is_main_process(env) is True
        assert D.is_distributed(env) is False

    def test_torchrun_env_is_detected(self):
        env = _torchrun_env(2)
        assert D.is_launched_by_torchrun(env) is True
        assert D.get_world_size(env) == 4

    def test_launcher_without_local_rank_falls_back_to_global(self):
        """`torch.distributed.launch` may omit LOCAL_RANK; the global rank is
        still a valid index, and guessing 0 would collide every rank on cuda:0.
        """
        env = {"RANK": "3", "WORLD_SIZE": "4"}
        assert D.get_local_rank(env) == 3

    def test_rank_is_read_from_the_process_group_when_one_exists(self, monkeypatch):
        """Once initialised, the process group is authoritative over the
        environment: a wrapper that rewrites RANK must not fool the sampler.
        """
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 5)
        monkeypatch.setattr(dist, "get_world_size", lambda: 8)
        assert D.is_distributed(_torchrun_env(0)) is True
        assert D.get_rank() == 5
        assert D.get_world_size() == 8
        assert D.get_local_rank() == 5
        assert D.is_main_process() is False

    def test_missing_distributed_support_is_a_raised_error_not_a_silent_skip(self, monkeypatch):
        monkeypatch.setattr(D, "is_distributed_available", lambda: False)
        with pytest.raises(RuntimeError, match="no torch.distributed support"):
            D.init_process_group(env=_torchrun_env(0))

    def test_missing_master_address_is_actionable(self):
        env = _torchrun_env(0)
        with pytest.raises(RuntimeError) as excinfo:
            D.init_process_group(env=env)
        message = str(excinfo.value)
        assert (
            "MASTER_ADDR" in message and "torchrun" in message
        ), f"the error must name the missing variable and the fix: {message}"

    def test_init_is_a_noop_outside_torchrun(self):
        assert D.init_process_group(env=_single_env()) == (0, 1, None)


class TestRankAwareSeed:
    def test_ranks_get_different_seeds(self):
        seeds = {D.rank_aware_seed(42, env=_torchrun_env(r)) for r in range(4)}
        assert seeds == {
            42,
            43,
            44,
            45,
        }, f"identical seeds make every rank compute the same gradient: {seeds}"

    def test_single_process_keeps_the_base_seed(self):
        assert D.rank_aware_seed(42, env=_single_env()) == 42

    def test_seed_is_reproducible_for_a_given_rank(self, monkeypatch):
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 2)
        monkeypatch.setattr(dist, "get_world_size", lambda: 4)
        first = D.rank_aware_seed(7)
        torch.rand(100)  # perturb the streams
        assert D.rank_aware_seed(7) == first

    def test_seeded_streams_actually_differ(self):
        draws = set()
        for rank in range(3):
            seed_everything(D.rank_aware_seed(11, env=_torchrun_env(rank)))
            draws.add(float(torch.rand(1)))
        assert len(draws) == 3, "rank-aware seeds produced identical torch streams"


class TestSamplerFactory:
    def test_none_outside_torchrun(self):
        assert D.make_distributed_sampler([1, 2, 3], env=_single_env()) is None

    def test_sampler_shards_across_ranks(self):
        samplers = [
            D.make_distributed_sampler(list(range(20)), env=_torchrun_env(r)) for r in range(4)
        ]
        assert all(s is not None for s in samplers)
        assert [s.num_replicas for s in samplers] == [4, 4, 4, 4]
        assert [s.rank for s in samplers] == [0, 1, 2, 3]
        assert (
            len({s.num_samples for s in samplers}) == 1
        ), "ranks disagree on the shard size, so the last batches differ"

    def test_sampler_partitions_the_dataset(self):
        seen = []
        for r in range(4):
            s = D.make_distributed_sampler(list(range(20)), env=_torchrun_env(r))
            seen.extend(s)
        assert sorted(seen) == list(range(20)), "DistributedSampler must partition, not copy"


class TestRankZeroOnlyAtomicWrites:
    @staticmethod
    def _save(path, payload):
        """Path-first save function, matching `src.train.save_checkpoint`.

        `torch.save` is (obj, path), so it cannot be handed to `save_on_rank0`
        directly — that inversion is exactly the kind of thing a caller gets
        wrong, so the wrapper is explicit here.
        """
        torch.save(payload, path)

    def test_only_rank_zero_saves(self, tmp_path):
        path = str(tmp_path / "ckpt.pth.tar")
        for r in range(4):
            D.save_on_rank0(self._save, path, {"rank": r}, env=_torchrun_env(r))
        assert torch.load(path, weights_only=True) == {"rank": 0}

    def test_no_temp_files_are_left_behind(self, tmp_path):
        path = str(tmp_path / "ckpt.pth.tar")
        D.save_on_rank0(self._save, path, {"w": torch.ones(2)}, env=_torchrun_env(0))
        assert os.listdir(tmp_path) == [
            "ckpt.pth.tar"
        ], f"temp files leaked: {os.listdir(tmp_path)}"

    def test_atomic_save_replaces_in_place(self, tmp_path):
        path = str(tmp_path / "x.bin")
        D.atomic_write_text(path, "first")
        D.atomic_write_text(path, "second")
        with open(path) as f:
            assert f.read() == "second"

    def test_atomic_save_never_leaves_a_truncated_file(self, tmp_path):
        """The old DDP script had every rank `torch.save` the same path. A reader
        between two ranks saw a partial zip, which `safe_torch_load` now
        correctly refuses — so the run stops instead of silently restarting.
        """
        path = str(tmp_path / "ckpt.pth.tar")
        D.atomic_torch_save({"a": torch.arange(1000)}, path)
        for _ in range(20):
            assert torch.load(path, weights_only=True)["a"].numel() == 1000

    def test_keyword_path_form_is_supported(self, tmp_path):
        path = str(tmp_path / "y.bin")
        D.save_on_rank0(
            lambda path, payload: D.atomic_write_text(path, payload),
            path=str(path),
            payload="kw",
            env=_torchrun_env(0),
        )
        with open(path) as f:
            assert f.read() == "kw"


class TestLogDirLock:
    @pytest.fixture()
    def _live_group(self, monkeypatch):
        """Pretend a process group exists without spawning anything."""
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    def test_lock_is_a_noop_outside_a_distributed_run(self, tmp_path):
        with D.log_dir_lock(str(tmp_path), env=_single_env()) as acquired:
            assert acquired is False
        assert os.listdir(tmp_path) == []

    def test_lock_is_exclusive_against_a_second_holder(self, tmp_path, _live_group):
        lock = str(tmp_path / D.LOCK_FILENAME)
        os.makedirs(tmp_path, exist_ok=True)
        with open(lock, "w") as f:  # simulate a live holder
            f.write("pid=1\n")
        with (
            pytest.raises(TimeoutError) as excinfo,
            D.log_dir_lock(str(tmp_path), timeout=0.05, poll=0.01),
        ):
            pass
        assert "logging.folder" in str(excinfo.value)

    def test_stale_after_is_not_tied_to_the_wait_timeout(self, tmp_path, _live_group):
        """Regression: sharing one timeout between 'wait for the holder' and
        'the holder is dead' made every wait end by stealing a live lock, so
        the primitive excluded nobody.
        """
        lock = str(tmp_path / D.LOCK_FILENAME)
        os.makedirs(tmp_path, exist_ok=True)
        with open(lock, "w") as f:
            f.write("pid=1\n")
        started = __import__("time").monotonic()
        with (
            pytest.raises(TimeoutError),
            D.log_dir_lock(str(tmp_path), timeout=0.05, poll=0.01, stale_after=3600),
        ):
            pass
        assert (
            __import__("time").monotonic() - started < 5.0
        ), "the lock must time out quickly, not hang for the stale window"
        assert os.path.exists(lock), "a live holder's lock must not be deleted"

    def test_stale_lock_is_broken(self, tmp_path, _live_group):
        lock = str(tmp_path / D.LOCK_FILENAME)
        os.makedirs(tmp_path, exist_ok=True)
        with open(lock, "w") as f:
            f.write("pid=999999\n")  # holder was killed mid-write
        old = __import__("time").time() - 10_000
        os.utime(lock, (old, old))
        with D.log_dir_lock(str(tmp_path), timeout=1.0, poll=0.01) as acquired:
            assert acquired is True
        assert not os.path.exists(lock), "the lock must be released on exit"


class TestBufferSyncDecision:
    def test_broadcast_buffers_is_off(self):
        assert D.BUFFER_SYNC_DECISION == "broadcast_buffers=False", (
            "broadcast_buffers=True makes DDP overwrite every rank's "
            "rdc.workspace_Q / wsd.target_Q with rank 0's, and the mechanisms "
            "read those buffers in the same forward they write them, so the "
            "consumed value is rank 0's from the previous step"
        )

    def test_find_unused_parameters_is_true(self):
        assert D.FIND_UNUSED_PARAMETERS is True, (
            "mechanism branches are step-dependent (pcr.level_gates, "
            "spc.freq_basis, cmc's second forward, the GAC second backward), so "
            "a False would crash on an unpredictable step schedule"
        )

    def _model(self):
        m = nn.Module()
        m.register_buffer("wsd_target_Q", torch.zeros(4, 2))
        m.register_buffer("wsd_target_cov", torch.zeros(4, 4))
        return m

    def test_consensus_targets_are_buffers_not_parameters(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.rdc = nn.Module()
                self.rdc.register_buffer("workspace_Q", torch.zeros(3, 2))
                self.wsd = nn.Module()
                self.wsd.register_buffer("target_Q", torch.zeros(3, 2))
                self.wsd.register_buffer("target_cov", torch.zeros(3, 3))
                self.jawp = nn.Module()
                self.jawp.workspace_Q = nn.Parameter(torch.zeros(3, 2))

        targets = D.mechanism_buffers_needing_consensus(Toy())
        names = [n for n, _ in targets]
        assert names == list(D.MECHANISM_BUFFERS_NEEDING_CONSENSUS)
        assert (
            "jawp.workspace_Q" not in names
        ), "jawp.workspace_Q is a Parameter and is already all-reduced by DDP"

    def test_all_reduce_averages_the_mechanism_buffers(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.rdc = nn.Module()
                self.rdc.register_buffer("workspace_Q", torch.zeros(2, 2))

        toy = Toy()
        toy.rdc.workspace_Q.fill_(4.0)
        calls = []

        def fake_all_reduce(tensor, op=None):
            calls.append(tensor.shape)
            tensor.mul_(2.0)  # pretend 2 ranks each contributed 4.0

        names = D.all_reduce_buffers(
            toy, world=2, all_reduce=fake_all_reduce, env=_torchrun_env(0, 2)
        )
        assert names == ["rdc.workspace_Q"]
        assert calls == [torch.Size([4])]
        assert torch.allclose(
            toy.rdc.workspace_Q, torch.full((2, 2), 4.0)
        ), "the all-reduced mean must land back in the buffer in place"

    def test_all_reduce_refuses_without_a_process_group(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.rdc = nn.Module()
                self.rdc.register_buffer("workspace_Q", torch.zeros(2, 2))

        with pytest.raises(RuntimeError, match="process group"):
            D.all_reduce_buffers(Toy(), env=_single_env())


class TestLossReductionAndUnwrap:
    def test_reduce_loss_is_identity_outside_ddp(self):
        assert D.reduce_loss(1.25, env=_single_env()) == 1.25

    def test_unwrap_model(self):
        inner = nn.Linear(2, 2)
        wrapped = DDPStub(inner)
        assert D.unwrap_model(wrapped) is inner
        assert D.unwrap_model(inner) is inner

    def test_cleanup_never_raises_without_a_group(self):
        D.cleanup()


class DDPStub:
    def __init__(self, module):
        self.module = module


class TestModuleIsImportInert:
    def test_importing_does_not_touch_torch_distributed_state(self):
        """A single-process run must be byte-identical whether or not this
        module has been imported: no process group, no env mutation, no
        `torch.distributed` import side effect at module scope.
        """
        import torch.distributed as dist

        assert dist.is_initialized() is False
        assert "RANK" not in os.environ
        assert D.get_rank() == 0 and D.get_world_size() == 1
        assert D.is_main_process() is True
