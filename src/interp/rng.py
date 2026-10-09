# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Seeded randomness for `src/interp/`, off the process-global torch stream.

Why this module exists
----------------------
`src/utils/seed.py` was never imported by anything under `src/interp/`, and 28
sites in this package drew from the **process-global** torch RNG. That is a
shared, mutable resource: two consumers interleave draws from it, so the value a
site sees is a function of what every other site did first. `tests/conftest.py`
makes the *suite* reproducible by reseeding per node id, but it cannot make a
site reproducible in isolation -- which is the property an ablation needs, since
an ablation reruns one function and compares across seeds.

The pattern this package already uses, and that this module centralises
----------------------------------------------------------------------
`src/models/cmc.py` (``_derive_mask_rng``) and ``src/models/collapse.py``
(``_derive_subsample_rng``) both solved this the same way, and both documented
why:

* a **private** ``torch.Generator`` rather than ``torch.manual_seed``, so the
  caller's stream never moves and a resume does not rewind it;
* the seed **derived** from ``torch.initial_seed()`` -- which is a *query*, not a
  draw, and reports the seed the process was given -- mixed with a per-site
  **salt** and a **stream index**, so two consumers on the same run seed cannot
  produce the same draws and a given (seed, stream) pair is a pure function;
* a **counter** rather than a persistent module-global generator, because a
  generator's position is not a tensor, never reaches ``state_dict``, and would
  silently replay the whole sequence from the start on resume.

That last property is why nothing here keeps a long-lived generator. It is also
why :func:`counter_generator` is documented as **not** resume-exact: it is the
convenience path for callers with no seed of their own, and a training loop
should pass ``step`` instead.

What is deliberately NOT here
----------------------------
No call to ``torch.manual_seed``, ``torch.cuda.manual_seed*``, or
``torch.random.set_rng_state``. The two sites in this package that used to save
and restore the global state around their draws now take a private generator,
which is strictly stronger: the save/restore dance was only correct if nothing
else drew in between, and it broke outright the moment a second draw ran on the
same thread.

Which resolver a site should use
-------------------------------
The choice is between "independent samples" and "one measurement", and getting
it wrong is not a style preference:

* :func:`generator_for` -- **idempotent**. Two calls with the same run seed
  return the same draws. Correct for anything whose output is *quoted as a
  number*: ``PolysemanticityIndex.compute``, a bootstrap CI, a permutation
  p-value. A metric that changes when you simply re-run it cannot be
  thresholded, and this module's own first draft got that wrong -- wiring the
  PSI index to :func:`counter_generator` made ``validate_polysemanticity()``
  return a different ``mean_psi`` on each call, which turned a knife-edge
  threshold into a coin flip.
* :func:`counter_generator` -- **distinct per call**. Correct for independent
  samples: a perturbation applied to a fresh input, a k-means restart. Wiring
  one of these to the idempotent resolver is the opposite bug, and a quieter
  one, because ``n_trials=3`` would then be the same trial three times.

Each consumer passes its own ``stream`` index. Streams are namespaced by
``site``, so ``"sae.resample"`` and ``"polysemanticity.kmeans"`` cannot collide
even when both are asked for draw 0.
"""

from __future__ import annotations

import torch

#: Mixed into every derived seed so this package's draws cannot coincide with
#: another consumer's draws from the same run seed (e.g. ``cmc._MASK_RNG_SALT``
#: or ``collapse._SUBSAMPLE_RNG_SALT``, which are separate constants for the
#: same reason).
INTERP_SALT = 0x1E7E9B1E

#: Seeds are reduced modulo this. ``torch.Generator.manual_seed`` accepts the
#: full int64 range, but staying inside the 31-bit band keeps the derived value
#: portable across torch versions and matches the modulus ``cmc`` and ``collapse``
#: already use.
_SEED_MODULUS = 2**31 - 1

#: Per-``site`` draw counter. Deliberately a plain module global and
#: deliberately NOT checkpointed: see the module docstring. A caller that needs
#: resume-exactness passes an explicit ``stream`` (e.g. a training step) and
#: never touches this.
_counters: dict[str, int] = {}


def run_seed() -> int:
    """The process run seed: what ``torch.manual_seed`` was last given.

    A *query*, not a draw -- ``torch.initial_seed`` reports the seed without
    reading randomness or advancing the stream. Callers that want a stream tied
    to the run rather than to a magic constant go through here; callers that
    want a stream tied to a published number pass their own ``seed``.

    Note that this returns whatever the *last* ``torch.manual_seed`` call set,
    so it is only meaningful when something seeded the process. Under
    ``pytest`` that is ``tests/conftest.py``; in a training run it is
    ``src/utils/seed.seed_everything``.
    """
    return int(torch.initial_seed())


def derive_generator(site: str, stream: int = 0) -> torch.Generator:
    """Build the generator for one draw, as a pure function of its inputs.

    Reproducible from ``(run seed, site, stream)`` alone: no state to persist,
    so a resume reconstructs the identical stream. Two different ``site``
    values never share a stream even at the same ``stream`` index.

    Args:
        site: namespaced stream label, e.g. ``"sae.resample"``. Only needs to
            be unique within this package.
        stream: which draw within that site -- a training step, a bootstrap
            replicate index. Mixing it in makes each draw independent of every
            other, so the whole sequence is reproducible from ``stream`` alone.

    Returns:
        A freshly seeded CPU generator.
    """
    base = run_seed() + INTERP_SALT
    derived = (base * 1_000_003 + int(stream) * 2_654_435_761) % _SEED_MODULUS
    gen = torch.Generator(device="cpu")
    gen.manual_seed(derived)
    return gen


def counter_generator(site: str) -> torch.Generator:
    """Next generator for ``site``: successive calls get *different* draws.

    This is the right resolver for a site whose draws are **independent
    samples** rather than one measurement -- a perturbation applied to a fresh
    input, a k-means restart. Calling it twice must not return the same numbers,
    or "n_trials=3" would silently become the same trial three times.

    It advances a process-level counter, so it is **not** resume-exact: a
    checkpoint cannot restore the counter, and a resume replays the sequence
    from the start. Prefer :func:`derive_generator` with a natural index when
    one exists (a training step, a trial number), and :func:`generator_for`
    when the caller has a seed.

    Args:
        site: namespaced stream label, same contract as
            :func:`derive_generator`.
    """
    return derive_generator(site, next_draw(site))


def next_draw(site: str) -> int:
    """Consume and return the next draw index for ``site``.

    Split out from :func:`counter_generator` so that a caller running a *loop*
    of draws can combine "which call is this" with "which iteration" and still
    get a distinct, reproducible stream per iteration:

        stream = rng.next_draw("my.site") + trial

    See :data:`_counters` for why this is process state and not checkpointed.
    """
    draw = _counters.get(site, 0)
    _counters[site] = draw + 1
    return draw


def generator_for(seed: int | None, site: str, stream: int = 0) -> torch.Generator:
    """Resolve a caller's optional ``seed`` into a usable private generator.

    The one place the two priorities are written down, so every entry point in
    this package resolves them the same way:

    1. ``seed`` given: a private generator seeded from ``seed`` mixed with
       ``stream``. Same seed and stream, same draws; the global stream is
       untouched, and two iterations of a loop differ from each other.
    2. ``seed is None``: :func:`derive_generator` on ``(site, stream)`` -- a
       pure function of the run seed.

    Both branches honour ``stream``, so a caller looping over trials gets an
    independent draw per trial whether or not the caller supplied a seed.

    Args:
        seed: caller-supplied seed, or None.
        site: namespaced stream label.
        stream: draw index within this call -- a trial number, a replicate
            index. A caller that wants successive calls to differ must pass an
            incrementing one; see :func:`next_draw`.

    Note the ``seed is None`` branch is deliberately **idempotent**: two calls
    with the same ``(run seed, site, stream)`` return the same generator. That
    is the right default for a *measurement* -- ``PolysemanticityIndex.compute``
    called twice on one input matrix must return the same number, because a
    metric that changes when you re-run it cannot be thresholded or compared.
    It is why this branch does not consult the counter.
    """
    if seed is None:
        return derive_generator(site, stream)
    gen = torch.Generator(device="cpu")
    gen.manual_seed((int(seed) * 1_000_003 + int(stream)) % _SEED_MODULUS)
    return gen
