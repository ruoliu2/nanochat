# Chain-of-Thought RLVR Implementation Plan

## Overview

This document outlines the implementation plan for **Chain-of-Thought (CoT) Reinforcement Learning with Verifiable Rewards (RLVR)** for the nanochat model, incorporating Meta Scale-RL best practices relevant to small models.

## Meta Scale-RL Best Practices (Small Model Focus)

### Practices We WILL Implement

| Practice | Description | Benefit for Small Models |
|----------|-------------|--------------------------|
| **RLVR (Verifiable Rewards)** | Binary rewards from programmatically verifiable tasks (math, code) | Clean supervision signal, no reward model needed |
| **On-Policy Training** | No PPO ratio/clip needed, simpler REINFORCE | Reduces complexity, faster training |
| **No KL Regularization** | Delete trust region/reference model penalty | Simplifies training, reference model not needed |
| **Advantage Normalization** | Use `(r - μ)` instead of z-score `(r - μ)/σ` | More stable for small sample sizes |
| **Token-Level Advantages (GAPO-style)** | Apply advantages at token level, not sequence level | Better credit assignment for long CoT |
| **FP32 Output Layer** | Compute logits in FP32 for numerical stability | Prevents NaN/inf issues in policy gradients |
| **Multiple Samples per Prompt** | Generate k samples per question for variance reduction | Better gradient estimates |
| **Zero-Variance Prompt Filtering** | Skip prompts that always succeed/fail | Focus compute on learnable examples |
| **Longer Generation Context** | Extended max_tokens for CoT reasoning chains | Allows model to "think" step-by-step |
| **Curriculum Learning** | Start with easier problems, progress to harder | Faster initial learning |

### Practices We Will SKIP (Large Model Only)

| Practice | Why Skip |
|----------|----------|
| Pipeline Parallelism (PP8) | Single GPU or small multi-GPU is sufficient |
| Async Training w/ Multiple Rollout Workers | Overkill for small model compute |
| Massive Batch Sizes (1000s) | Not needed, smaller batches work fine |
| Complex KL Annealing Schedules | On-policy removes need for this |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CoT RLVR Training Loop                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Sample Prompt from Task Pool                            │
│     ├── GSM8K (math with tool use)                          │
│     ├── MATH (competition math)                             │
│     └── Custom verifiable tasks                             │
│                                                             │
│  2. Generate K CoT Samples (with temperature)               │
│     ├── Extended max_tokens (512-1024 for reasoning)        │
│     ├── Tool use support (calculator)                       │
│     └── Track which tokens are sampled vs forced            │
│                                                             │
│  3. Compute Verifiable Rewards                              │
│     ├── Correctness: Extract final answer, verify           │
│     ├── Format Bonus: Reward #### marker usage              │
│     └── Length Penalty (optional): discourage verbosity     │
│                                                             │
│  4. Compute Advantages                                      │
│     ├── Group by prompt: advantage = r_i - mean(r_group)    │
│     └── No normalization by std (more stable)               │
│                                                             │
│  5. Policy Gradient Update                                  │
│     ├── loss = -sum(logp * advantage) / num_tokens          │
│     ├── Only train on sampled tokens (mask forced tokens)   │
│     └── FP32 for logits, BF16 for forward pass              │
│                                                             │
│  6. Zero-Variance Filtering (optional)                      │
│     └── Track prompt success rates, skip if too easy/hard   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Components

### 1. New Task: Math with CoT (`tasks/math_cot.py`)

Combines GSM8K with extended CoT reasoning format:
- Encourage step-by-step reasoning
- Verify final answer programmatically
- Support for calculator tool use

### 2. Reward Function Design

```python
def compute_reward(conversation, generated_text):
    """
    RLVR reward function with format bonuses.
    
    Returns:
        float: reward in [0, 1] range
    """
    # Base reward: correctness (0 or 1)
    is_correct = verify_answer(conversation, generated_text)
    reward = float(is_correct)
    
    # Format bonus: encourage structured output
    if "####" in generated_text:
        reward += 0.05  # small bonus for format compliance
    
    # Length penalty (optional, prevents reward hacking via verbosity)
    # reward -= 0.001 * max(0, len(generated_text) - 500)
    
    return min(max(reward, 0.0), 1.1)  # clamp
```

### 3. CoT Prompt Enhancement

System prompt to encourage reasoning:
```
Let's solve this step by step:
1. First, I'll identify what the problem is asking
2. Then I'll work through the calculation
3. Finally, I'll state the answer after ####
```

### 4. Zero-Variance Prompt Tracker

```python
class PromptTracker:
    """Track which prompts have zero variance (always right/wrong)."""
    
    def __init__(self, min_samples=8, skip_threshold=0.95):
        self.prompt_stats = {}  # prompt_id -> {correct: int, total: int}
        self.min_samples = min_samples
        self.skip_threshold = skip_threshold
    
    def should_skip(self, prompt_id):
        if prompt_id not in self.prompt_stats:
            return False
        stats = self.prompt_stats[prompt_id]
        if stats['total'] < self.min_samples:
            return False
        success_rate = stats['correct'] / stats['total']
        return success_rate > self.skip_threshold or success_rate < (1 - self.skip_threshold)
```

---

## Training Configuration

### Hyperparameters for Small Models

```python
# Generation
max_new_tokens = 512          # Extended for CoT reasoning
temperature = 1.0             # Encourage exploration
top_k = 50                    # Standard sampling

# RL
num_samples = 16              # Samples per prompt for variance reduction
examples_per_step = 8         # Prompts per gradient step
device_batch_size = 4         # Forward pass batch size (OOM protection)

# Learning Rate (Scale-RL: use lower LR for RL than SFT)
unembedding_lr = 0.002        # 0.5x SFT rate
embedding_lr = 0.1            # 0.5x SFT rate  
matrix_lr = 0.01              # 0.5x SFT rate

# Training
num_epochs = 3                # Multiple passes over GSM8K
save_every = 100              # Checkpoint frequency
eval_every = 50               # Evaluation frequency
```

### Learning Rate Schedule

Linear decay with warmup:
```python
def get_lr_multiplier(step, num_steps, warmup_frac=0.05):
    warmup_steps = int(num_steps * warmup_frac)
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0 - (step - warmup_steps) / (num_steps - warmup_steps)
```

---

## Evaluation Metrics

1. **Pass@1**: Accuracy with greedy decoding
2. **Pass@K**: At least one correct in K samples
3. **Reward Mean**: Average reward across samples
4. **Token Efficiency**: Correct answers / tokens generated

---

## Files

| File | Description |
|------|-------------|
| `scripts/chat_cot_rl.py` | Main CoT RLVR training script |
| `tasks/math_cot.py` | Math task with CoT-friendly formatting |
| `nanochat/prompt_tracker.py` | Zero-variance prompt filtering & curriculum |
| `nanochat/checkpoint_manager.py` | Updated to support `cot_rl` source |

---

## Training Guide

### Quick Start

```bash
# Train from SFT checkpoint (recommended starting point)
python -m scripts.chat_cot_rl --source=sft --run=my_cot_exp
```

### Loading Pre-trained Models

The `--source` parameter controls which checkpoint to start from:

| Source | Directory | Description |
|--------|-----------|-------------|
| `base` | `~/.cache/nanochat/base_checkpoints/` | Base pretrained model |
| `mid` | `~/.cache/nanochat/mid_checkpoints/` | Mid-training checkpoint |
| `sft` | `~/.cache/nanochat/chatsft_checkpoints/` | **SFT model (recommended)** |
| `rl` | `~/.cache/nanochat/chatrl_checkpoints/` | Previous RL checkpoint |
| `cot_rl` | `~/.cache/nanochat/cot_rl_checkpoints/` | Previous CoT-RL checkpoint |

### Training Commands

**Single GPU (for debugging or small experiments):**
```bash
python -m scripts.chat_cot_rl --source=sft --run=cot_exp1
```

**Multi-GPU (8x, for full training):**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_cot_rl -- \
    --source=sft \
    --run=cot_exp1 \
    --num_epochs=3
```

### Specifying Exact Checkpoints

```bash
# Use specific model tag (e.g., d12 for 12-layer model)
python -m scripts.chat_cot_rl --source=sft --model_tag=d12

# Use specific training step
python -m scripts.chat_cot_rl --source=sft --model_tag=d12 --step=500

# Continue from previous CoT-RL run
python -m scripts.chat_cot_rl --source=cot_rl --model_tag=d12 --step=200
```

### Key Hyperparameters

Override via CLI with `--param=value`:

```bash
python -m scripts.chat_cot_rl \
    --source=sft \
    --run=my_exp \
    --max_new_tokens=512 \      # Longer for reasoning (default: 512)
    --num_samples=16 \          # Samples per prompt (default: 16)
    --num_epochs=3 \            # Training epochs (default: 2)
    --matrix_lr=0.01 \          # Main LR (default: 0.01)
    --use_prompt_filtering=True \ # Zero-variance filtering (default: True)
    --use_curriculum=True       # Curriculum learning (default: True)
```

### Monitoring Training

With wandb enabled (`--run=<name>`), track:
- `pass@1`, `pass@4`, `pass@k`: Accuracy metrics
- `reward`: Average reward per step
- `seq_length`: Average generation length
- `tracker/num_skippable`: Prompts being filtered
- `lrm`: Learning rate multiplier

### Checkpoints

Models are saved to `~/.cache/nanochat/cot_rl_checkpoints/<model_tag>/`:
- Every `--save_every` steps (default: 100)
- At the final step

---

## Comparison: `chat_cot_rl.py` vs `chat_rl.py`

| Feature | `chat_rl.py` | `chat_cot_rl.py` |
|---------|--------------|------------------|
| Max tokens | 256 | 512 (for longer reasoning) |
| Learning rates | Full SFT rates | 0.5x (more stable) |
| LR warmup | ❌ | ✅ 5% warmup |
| Zero-variance filtering | ❌ | ✅ Skip easy/hard prompts |
| Curriculum learning | ❌ | ✅ Start with easier problems |
| Prompt tracking | ❌ | ✅ Per-prompt success rates |
| Task | GSM8K vanilla | MathCoT (CoT prompting) |
| Reward | Binary 0/1 | Binary + format bonus |

---

## Expected Improvements

Based on Scale-RL findings:
- **5-10% improvement** on GSM8K pass@1 over baseline RL
- **Faster convergence** due to variance reduction
- **More stable training** with FP32 output layer
- **Better generalization** from zero-variance filtering

---

## Troubleshooting

### OOM Errors
Reduce `device_batch_size`:
```bash
python -m scripts.chat_cot_rl --device_batch_size=2
```

### Slow Convergence
- Increase `num_samples` for better gradient estimates
- Disable curriculum if your model is already good: `--use_curriculum=False`

### Reward Not Improving
- Check if too many prompts are being filtered: look at `tracker/num_skippable` in wandb
- Reduce `prompt_filter_threshold` or disable: `--use_prompt_filtering=False`

### NaN/Inf Loss
The FP32 logits should prevent this, but if it happens:
- Lower learning rates: `--matrix_lr=0.005`
- Check your checkpoint isn't corrupted

---

## References

- Meta Scale-RL Paper (2024/2025)
- GRPO: Group Relative Policy Optimization
- GAPO: Token-level advantage estimation
- DeepSeek R1: CoT reasoning with RL
