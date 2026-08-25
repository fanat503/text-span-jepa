# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Train linear probes on frozen representations — NextLat pattern
# Usage: python train_probe.py --checkpoint path/to/checkpoint.pth.tar --probe linear

import argparse

import torch
from torch.utils.data import DataLoader

from src.datasets.kaggle import load_wikitext103
from src.eval.probes import FutureTokenProbe, GeometryMetrics
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.utils.torchio import safe_torch_load


def load_model_from_checkpoint(ckpt_path, config, device):
    """Load TextSpanJEPA from checkpoint file."""
    model = TextSpanJEPA(config).to(device)
    checkpoint = safe_torch_load(ckpt_path, map_location=device)

    if "encoder" in checkpoint:
        model.encoder.load_state_dict(checkpoint["encoder"])
    if "target_encoder" in checkpoint:
        model.target_encoder.load_state_dict(checkpoint["target_encoder"])
    if "predictor" in checkpoint:
        model.predictor.load_state_dict(checkpoint["predictor"])
    if "decoder" in checkpoint:
        model.decoder.load_state_dict(checkpoint["decoder"])

    print(f"Loaded checkpoint from {ckpt_path}")
    return model


def extract_representations(model, dataloader, device, max_batches=None):
    """Extract frozen representations from encoder."""
    model.eval()
    all_reprs = []
    all_ids = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
            else:
                input_ids = batch[0].to(device)

            h, _ = model.encoder(input_ids)
            # Mean-pool over sequence
            h_pooled = h.mean(dim=1)
            all_reprs.append(h_pooled.cpu())
            all_ids.append(input_ids.cpu())

    return torch.cat(all_reprs, dim=0), torch.cat(all_ids, dim=0)


def probe_geometry(representations):
    """Compute geometry metrics on representations."""
    metrics = GeometryMetrics.compute(representations)
    print("\n=== Representation Geometry ===")
    for k, v in sorted(metrics.items()):
        if k != "error":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train probes on frozen Text-Span JEPA representations"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML (overrides checkpoint config)"
    )
    parser.add_argument(
        "--probe",
        type=str,
        default="geometry",
        choices=["linear", "future", "geometry", "all"],
        help="Type of probe to run",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--max-batches", type=int, default=100, help="Max batches for representation extraction"
    )
    parser.add_argument("--data-dir", type=str, default="data/wikitext-103")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load config
    if args.config:
        import yaml

        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        model_cfg = {
            **cfg.get("model", {}),
            "vocab_size": 50257,
            "max_seq_len": cfg.get("data", {}).get("max_seq_len", 512),
        }
        config = TextSpanJEPAConfig(**model_cfg)
    else:
        config = TextSpanJEPAConfig()

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, config, device)

    # Load data
    dataset, tokenizer = load_wikitext103(
        seq_len=config.max_seq_len,
        split="valid",
        data_dir=args.data_dir,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Extract representations
    print("Extracting representations...")
    reprs, _ids = extract_representations(model, dataloader, device, max_batches=args.max_batches)
    print(f"Extracted {reprs.shape[0]} representations of dim {reprs.shape[1]}")

    if args.probe in ("geometry", "all"):
        probe_geometry(reprs)

    if args.probe in ("linear", "all"):
        print("\n=== Linear Probe ===")
        print("(requires labeled data — skipping for unsupervised pretraining)")
        print("For downstream eval, provide a classification dataset and use src/eval/probes.py")

    if args.probe in ("future", "all"):
        print("\n=== Future Token Probe ===")
        probe = FutureTokenProbe(
            embed_dim=config.embed_dim,
            vocab_size=tokenizer.vocab_size,
            offsets=tuple(config.future_offsets),
        )
        # Quick eval on subset
        results = probe.evaluate(model, dataset, device=device, max_steps=50)
        for k, v in sorted(results.items()):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
