# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Kaggle dataset loading: WikiText-103 / BookCorpus / C4 small
# Works on Kaggle notebooks with GPU T4/P100

import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer


class TextDataset(Dataset):
    """Tokenized text dataset for self-supervised pretraining.

    Args:
        token_ids: list or array of token indices
        seq_len: sequence length for each chunk
        drop_last: if True, discard incomplete final chunk (default: True,
            matching DataLoader drop_last behavior for training).
            Set to False for evaluation to use all data.
        pad_id: padding token id for incomplete final chunk (default: 0)
    """

    def __init__(self, token_ids, seq_len=512, drop_last=True, pad_id=0):
        self.seq_len = seq_len
        self.chunks = []
        for i in range(0, len(token_ids) - seq_len + 1, seq_len):
            self.chunks.append(token_ids[i : i + seq_len])
        # Handle incomplete final chunk
        if not drop_last:
            remainder_start = len(self.chunks) * seq_len
            if remainder_start < len(token_ids):
                remainder = token_ids[remainder_start:]
                padded = remainder + [pad_id] * (seq_len - len(remainder))
                self.chunks.append(padded)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.chunks[idx], dtype=torch.long)}


def load_wikitext103(
    tokenizer_name="gpt2", seq_len=512, split="train", data_dir="/kaggle/input/wikitext-103"
):
    """Load WikiText-103 dataset for Kaggle."""
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    file_map = {
        "train": "wiki.train.tokens",
        "valid": "wiki.valid.tokens",
        "test": "wiki.test.tokens",
    }

    filepath = None
    for root, dirs, files in os.walk(data_dir):
        target = file_map.get(split)
        if target in files:
            filepath = os.path.join(root, target)
            break

    if filepath is None:
        try:
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
            text = "\n".join(ds["text"])
        except Exception:
            raise FileNotFoundError(
                f"Could not find WikiText-103 {split} data in {data_dir}. "
                f"Add the wikitext-103 dataset to your Kaggle notebook."
            )
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

    token_ids = tokenizer.encode(text)
    print(f"Loaded WikiText-103 {split}: {len(token_ids):,} tokens")
    # Pad partial final chunks with the tokenizer's real pad id (GPT-2 id 0
    # is a live "!" token; treating it as padding corrupts masking statistics).
    chunk_pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return TextDataset(token_ids, seq_len=seq_len, pad_id=chunk_pad_id), tokenizer


def load_bookcorpus(tokenizer_name="gpt2", seq_len=512, data_dir="/kaggle/input/bookcorpus"):
    """Load BookCorpus subset for Kaggle."""
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    try:
        from datasets import load_dataset

        ds = load_dataset("bookcorpus", split="train", streaming=True)
        all_tokens = []
        count = 0
        for item in ds:
            all_tokens.extend(tokenizer.encode(item["text"]))
            count += 1
            if count >= 10000:
                break
        print(f"Loaded BookCorpus subset: {len(all_tokens):,} tokens from {count} books")
        return TextDataset(all_tokens, seq_len=seq_len), tokenizer
    except Exception as e:
        print(f"Could not load BookCorpus: {e}")
        print("Falling back to WikiText-103")
        return load_wikitext103(tokenizer_name, seq_len, "train", data_dir)


def make_dataloader(
    dataset,
    batch_size=64,
    num_workers=2,
    shuffle=True,
    worker_init_fn=None,
    generator=None,
):
    """Create DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def get_mask_token_id(tokenizer):
    """Get mask token ID for the tokenizer."""
    if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    return tokenizer.eos_token_id
