# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Tests for CLI overrides: `--seed` must win, seed 0 must survive, no mutation."""

from __future__ import annotations

from src.utils.config_overrides import apply_cli_overrides


def _config():
    return {
        "seed": 7,
        "meta": {"seed": 3, "model_name": "text_span_jepa"},
        "logging": {"folder": "runs/original"},
    }


class TestSeedOverride:
    def test_cli_seed_wins_over_both_config_fields(self):
        resolved = apply_cli_overrides(_config(), seed=11)
        # train.main reads meta.seed before top-level seed; writing it there
        # makes the CLI value win without changing that documented order.
        assert resolved["meta"]["seed"] == 11
        assert resolved["seed"] == 7

    def test_seed_zero_is_applied(self):
        resolved = apply_cli_overrides(_config(), seed=0)
        assert resolved["meta"]["seed"] == 0

    def test_none_leaves_the_config_seed_alone(self):
        resolved = apply_cli_overrides(_config(), seed=None)
        assert resolved["meta"]["seed"] == 3

    def test_meta_is_created_when_absent(self):
        resolved = apply_cli_overrides({"seed": 2}, seed=5)
        assert resolved["meta"]["seed"] == 5


class TestOutputDirOverride:
    def test_output_dir_sets_logging_folder(self):
        resolved = apply_cli_overrides(_config(), output_dir="runs/jepa_s1")
        assert resolved["logging"]["folder"] == "runs/jepa_s1"

    def test_logging_is_created_when_absent(self):
        resolved = apply_cli_overrides({}, output_dir="runs/x")
        assert resolved["logging"]["folder"] == "runs/x"


class TestImmutability:
    def test_input_config_is_not_mutated(self):
        base = _config()
        resolved = apply_cli_overrides(base, seed=11, output_dir="runs/new")
        assert base["meta"]["seed"] == 3
        assert base["logging"]["folder"] == "runs/original"
        assert resolved is not base
        assert resolved["meta"] is not base["meta"]

    def test_both_overrides_at_once(self):
        resolved = apply_cli_overrides(_config(), seed=9, output_dir="runs/jepa_s9")
        assert resolved["meta"]["seed"] == 9
        assert resolved["logging"]["folder"] == "runs/jepa_s9"
