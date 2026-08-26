# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""End-to-end training-loop gates (audit R18).

Executes the REAL `train.main()` — datapipe, masking, full mechanism stack,
CMC second-forward bridge, GAC adaptation hook, STA live signal, WSR lagged
capture, SPC retraction, checkpointing, retention pruning, resume replay —
against a stubbed WikiText loader. This is the surface that hid the R1
critical entry-point crash for its entire lifetime.
"""

import os

import torch
import pytest

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.datasets.kaggle import TextDataset
from src.train import load_checkpoint, main, save_checkpoint
from src.utils.seed import seed_everything

VOCAB = 64
SEQ = 16
TOKENS = 512  # -> ipe = TOKENS // 4 = 128 batches @ bs=2


class _StubTokenizer:
    vocab_size = VOCAB
    pad_token_id = 0
    eos_token_id = 0
    mask_token_id = None


def _fake_load(split="train", seq_len=SEQ, **_kw):
    torch.manual_seed(0 if split == "train" else 1)
    n = TOKENS // 2 if split == "train" else TOKENS // 4
    tokens = torch.randint(1, VOCAB, (n * seq_len,)).tolist()
    return TextDataset(tokens, seq_len=seq_len), _StubTokenizer()


def _config(folder, epochs, load_checkpoint=False):
    return {
        "meta": {
            "seed": 42,
            "model_name": "text_span_jepa",
            "use_bfloat16": False,
            "load_checkpoint": load_checkpoint,
        },
        "data": {
            "batch_size": 2,
            "max_seq_len": SEQ,
            "num_workers": 0,
            "mask_ratio": 0.3,
            "span_length_range": [2, 4],
        },
        "model": {
            "embed_dim": 32,
            "encoder_depth": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "predictor_embed_dim": 16,
            "predictor_depth": 1,
            "future_offsets": [1],
            "num_refine_steps": 1,
            "future_warmup_steps": 0,
            "lambda_span": 1.0,
            "lambda_future": 0.2,
            "lambda_decoder": 0.1,
            "use_jawp": True,
            "jawk_k_start": 2,
            "jawk_k_end": 4,
            "jawk_curriculum_steps": 0,
            "jawk_init": "random",
            "use_swip": True,
            "lambda_swip": 0.02,
            "use_sta": True,
            "lambda_sta": 0.05,
            "sta_warmup_steps": 0,
            "use_wsd": True,
            "lambda_wsd": 0.05,
            "use_puc": True,
            "lambda_puc": 0.05,
            "puc_warmup_steps": 0,
            "use_rdc": True,
            "lambda_rdc": 0.05,
            "use_spc": True,
            "lambda_spc": 0.02,
            "use_cgn": True,
            "use_gac": True,
            "lambda_gac": 0.05,
            "gac_tau_grad": 1e-5,
            "gac_warmup_steps": 0,
            "use_cmc": True,
            "lambda_cmc": 0.05,
            "cmc_interval": 1,
            "cmc_min_overlap_ratio": 0.05,
            "use_wsr": True,
            "lambda_wsr": 0.05,
            "wsr_mode": "gradient",
            "wsr_warmup_steps": 0,
        },
        "optimization": {
            "epochs": epochs,
            "lr": 1e-3,
            "start_lr": 1e-4,
            "final_lr": 1e-5,
            "warmup": 1,
            "weight_decay": 0.0,
            "final_weight_decay": 0.0,
            "grad_accum_steps": 1,
        },
        "logging": {
            "folder": str(folder),
            "log_freq": 1000,
            "keep_last_epoch_ckpts": 1,
        },
    }


def _global_step(folder):
    ckpt = torch.load(
        os.path.join(str(folder), "checkpoint-latest.pth.tar"),
        map_location="cpu",
        weights_only=False,
    )
    return ckpt["global_step"]


@pytest.fixture()
def _patch_dataset(monkeypatch):
    monkeypatch.setattr("src.datasets.kaggle.load_wikitext103", _fake_load)


class TestEndToEndTrainingLoop:
    def test_two_epochs_train_save_prune(self, tmp_path, _patch_dataset):
        main(_config(tmp_path, epochs=2))
        d = tmp_path
        assert (d / "checkpoint-latest.pth.tar").exists()
        assert (d / "checkpoint-ep2.pth.tar").exists(), "final epoch ckpt must exist"
        assert not (
            d / "checkpoint-ep1.pth.tar"
        ).exists(), "keep_last_epoch_ckpts=1 must prune older epoch checkpoints"
        assert (d / "best.pt").exists(), "validation ran, best.pt expected"
        assert (d / "params-text-span-jepa.yaml").exists()
        assert (d / "train_log.csv").exists()
        assert _global_step(d) == 2 * (TOKENS // 4)

    def test_resume_continues_global_step(self, tmp_path, _patch_dataset):
        cfg = _config(tmp_path, epochs=2)
        main(cfg)
        gs_before = _global_step(tmp_path)

        cfg3 = _config(tmp_path, epochs=3, load_checkpoint=True)
        main(cfg3)
        gs_after = _global_step(tmp_path)
        ipe = TOKENS // 4
        assert gs_before == 2 * ipe
        assert gs_after == 3 * ipe, "resume must continue, not restart, the step counter"
        assert (tmp_path / "checkpoint-ep3.pth.tar").exists()


class TestCheckpointRoundTrip:
    def test_cgn_state_survives_round_trip(self, tmp_path):
        """Regression (R18 advisory): R12 dropped the tau-anneal counter from
        checkpoint saves while removing the masked-logit table; every resume
        silently reset CGN's temperature schedule. No suite gate exercised a
        CGN-enabled round trip until this test.
        """
        seed_everything(11)
        cfg = TextSpanJEPAConfig(
            embed_dim=32,
            encoder_depth=1,
            num_heads=2,
            predictor_embed_dim=16,
            predictor_depth=1,
            use_cgn=True,
            cgn_anneal_steps=100,
        )
        model = TextSpanJEPA(cfg)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        path = str(tmp_path / "cgn-ckpt.pth.tar")

        model.cgn.total_steps.fill_(5000)
        vis_before = model.cgn.gate_logits_visible.detach().clone()

        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            epoch=0,
            global_step=5000,
            model_name="text_span_jepa",
        )

        # Corrupt in-memory state to prove LOAD is what restores it.
        model.cgn.total_steps.fill_(7)
        model.cgn.gate_logits_visible.data.fill_(99.0)

        load_checkpoint(path, model, optimizer, scaler, model_name="text_span_jepa")

        assert int(model.cgn.total_steps.item()) == 5000
        assert torch.equal(
            model.cgn.gate_logits_visible.detach(), vis_before
        ), "gate logits must survive the checkpoint round trip"
