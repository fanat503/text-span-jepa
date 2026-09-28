# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""The single, greppable gate for mechanism state mutation.

``src/train.py::_validate`` runs every mechanism's loss under
``model.eval()`` + ``torch.no_grad()``. Several mechanism modules nevertheless
wrote their EMA / step buffers on that path, so the *trained weights* depended
on whether a validation split was loaded: two machines with the same seed
produced different models. Measured before the fix: 24 buffers mutated under
``eval()``.

THE RULE (do not bypass): a mechanism may only mutate a registered buffer or a
persistent statistic while ``self.training`` is True. Route every such write
through ``self._mutate_state`` so that

    grep -n "_mutate_state" src/models/*.py

lists every state write in the codebase. Under ``eval()`` the callable is not
invoked at all, which makes the forward a pure function of its inputs.

Guarding only is not enough to prove the guard holds, so the test suite also
asserts that training mode still mutates exactly the buffers it used to --
a guard that over-corrects and freezes training is as broken as no guard.
"""

from __future__ import annotations

from torch import nn

__all__ = ["TrainingStateGuard"]


class TrainingStateGuard(nn.Module):
    """Mixin enforcing that state mutation happens only in training mode.

    Mechanism modules inherit this instead of ``nn.Module`` directly. It adds no
    parameters and no behaviour in training mode; it only decides whether a
    state write is invoked at all.

    Args:
        args: forwarded to ``nn.Module.__init__``.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _mutate_state(self, fn, *args, **kwargs):
        """Run ``fn(*args, **kwargs)`` only while training.

        Returns whatever ``fn`` returns in training mode, and ``None`` under
        ``eval()`` (where it is not called).

        Args:
            fn: a bound method or zero-argument callable that writes buffers
                or persistent statistics.
            *args: positional arguments forwarded to ``fn``.
            **kwargs: keyword arguments forwarded to ``fn``.

        """
        if not self.training:
            return None
        return fn(*args, **kwargs)
