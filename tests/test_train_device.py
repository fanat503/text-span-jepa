# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Device-resolution gates. No GPU, no TPU, no `torch.distributed` required.

`src/train.py` used `torch.device("cuda" if cuda.is_available() else "cpu")`,
which is wrong in three separate ways:

* `pytorch_xla` is not consulted, so a TPU node trains on CPU;
* `LOCAL_RANK` is ignored, so under `torchrun` every rank resolves to
  `cuda:0` and the ranks fight over one device;
* there is no error at all when `PJRT_DEVICE` is set but XLA is missing, so the
  run dies with an `AttributeError` from inside a training step, after the
  dataset has already been loaded.

`resolve_device` takes the environment mapping and the two availability probes
as arguments, which makes the whole branch structure testable on a CPU-only
box.
"""

import pytest
import torch

from src.train import (
    XLA_INSTALL_COMMAND,
    resolve_device,
    xla_is_available,
)

CPU = {"env": {}, "xla_available": lambda: False, "cuda_available": lambda: False}


def _cuda():
    return {"env": {}, "xla_available": lambda: False, "cuda_available": lambda: True}


class TestTpuBranch:
    def test_pjrt_device_with_xla_available_selects_xla(self):
        dev = resolve_device(
            env={"PJRT_DEVICE": "TPU"},
            xla_available=lambda: True,
            cuda_available=lambda: False,
        )
        assert dev.type == "xla", "PJRT_DEVICE set and XLA importable must select XLA"

    def test_pjrt_device_without_xla_raises_actionable_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            resolve_device(
                env={"PJRT_DEVICE": "TPU"},
                xla_available=lambda: False,
                cuda_available=lambda: True,
            )
        message = str(excinfo.value)
        assert (
            XLA_INSTALL_COMMAND in message
        ), f"the error must name the install command; got: {message}"
        assert "PJRT_DEVICE" in message
        assert "TPU" in message, "the error must name the offending device value"

    def test_xla_beats_cuda(self):
        """TPU wins over a visible GPU: a TPU node often has CUDA visible too."""
        dev = resolve_device(
            env={"PJRT_DEVICE": "TPU"},
            xla_available=lambda: True,
            cuda_available=lambda: True,
        )
        assert dev.type == "xla"

    def test_pjrt_device_outranks_local_rank(self):
        dev = resolve_device(
            env={"PJRT_DEVICE": "CPU", "LOCAL_RANK": "0", "WORLD_SIZE": "4"},
            xla_available=lambda: True,
            cuda_available=lambda: True,
        )
        assert dev.type == "xla"

    def test_xla_is_available_requires_both_pjrt_and_importable(self, monkeypatch):
        monkeypatch.setenv("PJRT_DEVICE", "TPU")
        monkeypatch.setattr("src.train.importlib.util.find_spec", lambda name: None)
        assert xla_is_available() is False, (
            "PJRT_DEVICE alone is not enough: without pytorch_xla the run must "
            "raise the install error, not select a device it cannot use"
        )
        monkeypatch.setenv("PJRT_DEVICE", "")
        assert (
            xla_is_available() is False
        ), "pytorch_xla importable without a PJRT device must not hijack a CPU box"


class TestDdpGpuBranch:
    def test_local_rank_selects_that_gpu(self):
        dev = resolve_device(
            env={"LOCAL_RANK": "2", "WORLD_SIZE": "4"},
            xla_available=lambda: False,
            cuda_available=lambda: True,
        )
        assert dev.type == "cuda"
        assert dev.index == 2, f"torchrun rank 2 must land on cuda:2, not cuda:0; got {dev}"

    def test_every_rank_gets_a_distinct_device(self):
        devices = {
            str(
                resolve_device(
                    env={"LOCAL_RANK": str(r), "WORLD_SIZE": "4"},
                    xla_available=lambda: False,
                    cuda_available=lambda: True,
                )
            )
            for r in range(4)
        }
        assert devices == {
            "cuda:0",
            "cuda:1",
            "cuda:2",
            "cuda:3",
        }, f"ranks collided on one device: {sorted(devices)}"

    def test_local_rank_zero_is_honoured_not_treated_as_absent(self):
        dev = resolve_device(
            env={"LOCAL_RANK": "0", "WORLD_SIZE": "1"},
            xla_available=lambda: False,
            cuda_available=lambda: True,
        )
        assert dev.type == "cuda" and dev.index == 0

    def test_non_integer_local_rank_raises(self):
        with pytest.raises(RuntimeError, match="LOCAL_RANK"):
            resolve_device(
                env={"LOCAL_RANK": "not-a-number"},
                xla_available=lambda: False,
                cuda_available=lambda: True,
            )

    def test_local_rank_without_cuda_still_names_the_gpu(self, caplog):
        """Fail-fast at tensor allocation, where the error is meaningful, rather
        than silently demoting every rank to cpu:0 and producing N identical
        "independent" replicas.
        """
        with caplog.at_level("WARNING"):
            dev = resolve_device(
                env={"LOCAL_RANK": "1"},
                xla_available=lambda: False,
                cuda_available=lambda: False,
            )
        assert str(dev) == "cuda:1"
        assert any("LOCAL_RANK" in r.message for r in caplog.records)


class TestCudaAndCpuFallbacks:
    def test_nothing_set_is_cpu(self):
        assert resolve_device(**CPU) == torch.device("cpu")

    def test_cuda_available_is_cuda_zero(self):
        dev = resolve_device(**_cuda())
        assert dev.type == "cuda" and dev.index == 0

    def test_default_dependencies_do_not_explode(self):
        """The zero-argument form used by `main()` must work on a CPU box."""
        assert resolve_device().type in ("cpu", "cuda", "xla")


class TestPriorityChain:
    def test_documented_order(self):
        """TPU > DDP-GPU > CUDA > CPU, asserted end to end."""
        assert (
            resolve_device(
                env={"PJRT_DEVICE": "TPU", "LOCAL_RANK": "1"},
                xla_available=lambda: True,
                cuda_available=lambda: True,
            ).type
            == "xla"
        )
        assert resolve_device(
            env={"LOCAL_RANK": "1"},
            xla_available=lambda: False,
            cuda_available=lambda: True,
        ) == torch.device("cuda", 1)
        assert resolve_device(**_cuda()) == torch.device("cuda", 0)
        assert resolve_device(**CPU) == torch.device("cpu")

    def test_empty_env_value_is_not_a_tpu_signal(self):
        assert resolve_device(
            env={"PJRT_DEVICE": "", "LOCAL_RANK": "3"},
            xla_available=lambda: False,
            cuda_available=lambda: True,
        ) == torch.device("cuda", 3)
