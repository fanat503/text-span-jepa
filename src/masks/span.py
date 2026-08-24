# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Span masking: contiguous block masking for text sequences
# Adapted from I-JEPA multiblock masking + SpanBERT span selection

import numpy as np
import torch


class SpanMaskCollator:
    """Span masking collator for Text-Span JEPA.

    Generates contiguous span masks (not random token masks) to force
    the model to use broader context. Supports mask curriculum.
    """

    def __init__(
        self,
        mask_ratio=0.35,
        span_length_range=(3, 10),
        max_num_spans=None,
        mask_token_id=0,
        mask_ratio_start=None,
        mask_ratio_end=None,
        curriculum_steps=0,
        pad_id=0,
    ):
        self.mask_ratio = mask_ratio
        self.span_length_range = span_length_range
        self.max_num_spans = max_num_spans
        self.mask_token_id = mask_token_id
        self.pad_id = pad_id
        # Explicit None checks (not `or`): explicit 0.0 start/end must win
        # over the static mask_ratio fallback.
        self.mask_ratio_start = mask_ratio_start if mask_ratio_start is not None else mask_ratio
        self.mask_ratio_end = mask_ratio_end if mask_ratio_end is not None else mask_ratio
        self.curriculum_steps = curriculum_steps
        self._step = 0

    def step(self):
        self._step += 1

    @property
    def current_mask_ratio(self):
        if self.curriculum_steps <= 0:
            return self.mask_ratio
        progress = min(self._step / self.curriculum_steps, 1.0)
        return self.mask_ratio_start + progress * (self.mask_ratio_end - self.mask_ratio_start)

    def generate_mask(self, seq_len):
        """Generate a span mask for a single sequence."""
        mask = np.zeros(seq_len, dtype=np.int32)
        target_num_masked = int(seq_len * self.current_mask_ratio)
        num_masked = 0
        max_spans = self.max_num_spans or (seq_len // self.span_length_range[0])

        for _ in range(max_spans):
            if num_masked >= target_num_masked:
                break
            span_len = np.random.randint(self.span_length_range[0], self.span_length_range[1] + 1)
            start = np.random.randint(0, max(1, seq_len - span_len + 1))
            end = min(start + span_len, seq_len)
            for i in range(start, end):
                if mask[i] == 0 and num_masked < target_num_masked:
                    mask[i] = 1
                    num_masked += 1
        return mask

    def __call__(self, batch):
        """Collate a batch and generate span masks."""
        if isinstance(batch[0], dict):
            input_ids_list = [item["input_ids"] for item in batch]
        else:
            input_ids_list = batch

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [
                (
                    x.clone().detach()
                    if isinstance(x, torch.Tensor)
                    else torch.tensor(x, dtype=torch.long)
                )
                for x in input_ids_list
            ],
            batch_first=True,
            padding_value=0,
        )
        B, T = input_ids.shape

        masks = []
        for i in range(B):
            non_pad = (input_ids[i] != self.pad_id).sum().item()
            mask = self.generate_mask(non_pad)
            full_mask = np.zeros(T, dtype=np.int32)
            full_mask[:non_pad] = mask
            masks.append(full_mask)

        mask_positions = torch.tensor(np.stack(masks), dtype=torch.long)
        original_input_ids = input_ids.clone()
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_positions.bool()] = self.mask_token_id

        # NOTE: step() is NOT called here — the training loop calls
        # mask_collator.step() after each training step to advance the
        # curriculum. Calling step() both here and in the training loop
        # would double-advance the curriculum.

        return {
            "masked_input_ids": masked_input_ids,
            "original_input_ids": original_input_ids,
            "mask_positions": mask_positions,
        }
