# Training Optimizations

This document describes two performance optimizations for nanochat training:
1. **Sequence Packing** — Eliminates wasted compute at document boundaries
2. **FP8 Training** — Leverages H100 FP8 tensor cores for ~2× speedup

---

## 1. Sequence Packing

### Problem

The current dataloader concatenates documents into a continuous token stream:

```
[Doc1 tokens...][Doc2 tokens...][Doc3 tokens...]
     ^--- batch boundary might split here
```

When a batch boundary falls in the middle of a document, no tokens are wasted. But when we use the standard next-token prediction loss, we're predicting the first token of Doc2 given the last token of Doc1 — this is noise that hurts training.

### Solution

Track document boundaries and mask out the loss at those positions:

```
Tokens:  [... doc1_end] [doc2_start ...]
Targets: [... doc2_start] [doc2_token2 ...]
Mask:    [... 1        ] [0           ] [1 ...]
                          ^-- mask out this prediction
```

### Implementation

**Files modified:**
- `nanochat/dataloader.py` — Add `packing_dataloader_with_state()` function

**Key changes:**
1. Return a `loss_mask` tensor alongside `inputs` and `targets`
2. Mark positions where target is from a different document with `mask=0`
3. Use `ignore_index=-1` in loss computation for masked positions

**Usage:**
```python
# In base_train.py
train_loader = packing_dataloader_with_state(B, T, split="train", ...)
x, y, loss_mask, state = next(train_loader)
loss = model(x, y, loss_mask=loss_mask)
```

**Expected speedup:** 10-30% depending on average document length vs sequence length.

---

## 2. FP8 Training

### Problem

H100 GPUs have dedicated FP8 tensor cores that provide ~2× the FLOPS of BF16, but nanochat currently only uses BF16.

### Solution

Use NVIDIA Transformer Engine to:
1. Replace `nn.Linear` layers with `te.Linear` (FP8-capable)
2. Wrap the forward pass with `te.fp8_autocast()`

### Requirements

- NVIDIA H100 (or newer) GPU
- CUDA 12.0+
- Transformer Engine library: `pip install transformer-engine`

### Implementation

**Files modified:**
- `nanochat/gpt.py` — Add `GPTFP8` model variant
- `scripts/base_train.py` — Add `--fp8` flag

**Key changes:**

1. **New model class `GPTFP8`:**
   - Uses `te.Linear` instead of `nn.Linear` for all projections
   - Attention, MLP, and output projection all use FP8

2. **FP8 autocast context:**
   ```python
   import transformer_engine.pytorch as te
   
   with te.fp8_autocast(enabled=True):
       loss = model(x, y)
   ```

3. **Graceful fallback:**
   - If Transformer Engine not installed, fall back to BF16
   - If not on H100, fall back to BF16 with warning

**Usage:**
```bash
# Enable FP8 training
torchrun --nproc_per_node=8 -m scripts.base_train --fp8=True
```

**Expected speedup:** 30-50% on H100s.

---

## Combined Usage

Both optimizations can be used together:

```bash
torchrun --nproc_per_node=8 -m scripts.base_train \
    --fp8=True \
    --use_packing=True
```

---

## Implementation Status

| Feature | Status | Files |
|---------|--------|-------|
| Sequence Packing | ✅ Implemented | `dataloader.py` |
| FP8 Training | ✅ Implemented | `gpt.py`, `base_train.py` |

---

## Benchmarks

*To be filled in after testing on H100 cluster.*

| Config | Tokens/sec | MFU | vs Baseline |
|--------|------------|-----|-------------|
| Baseline (BF16, no packing) | - | - | 1.0× |
| + Sequence Packing | - | - | ~1.2× |
| + FP8 | - | - | ~1.5× |
| + Both | - | - | ~1.8× |
