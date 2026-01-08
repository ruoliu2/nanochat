"""
Chain-of-Thought RLVR Training with Meta Scale-RL Best Practices.

This script implements reinforcement learning for reasoning tasks using
verifiable rewards (RLVR) with the following Scale-RL best practices:

Best Practices Implemented:
1. RLVR - Binary rewards from programmatically verifiable tasks
2. On-policy training - No PPO ratio/clip, simpler REINFORCE
3. No KL regularization - Trust region not needed for on-policy
4. Advantage normalization - (r - μ) instead of z-score
5. Token-level advantages (GAPO-style) - Better credit assignment
6. FP32 output layer - Numerical stability in policy gradients
7. Multiple samples per prompt - Variance reduction
8. Zero-variance prompt filtering - Focus compute on learnable examples
9. Longer generation context - Extended max_tokens for CoT
10. Curriculum learning - Start with easier problems

Best Practices Skipped (large model only):
- Pipeline parallelism (PP8)
- Async training with multiple rollout workers
- Massive batch sizes

Usage:
    Single GPU:
        python -m scripts.chat_cot_rl

    Multi-GPU:
        torchrun --standalone --nproc_per_node=8 -m scripts.chat_cot_rl -- --run=cot_rl_exp1
"""

import os
import itertools
import wandb
import torch
import torch.distributed as dist

from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    get_base_dir,
    DummyWandb,
)
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from nanochat.prompt_tracker import PromptTracker, CurriculumScheduler
from tasks.math_cot import MathCoT

# -----------------------------------------------------------------------------
# RLVR Hyperparameters with Scale-RL best practices

run = "dummy"  # wandb run name ("dummy" = no logging)
source = "sft"  # base|mid|sft - which checkpoint to load
model_tag = None  # model tag to load (None = auto-detect largest)
step = None  # checkpoint step to load (None = latest)
dtype = "bfloat16"

# Generation settings (extended for CoT)
max_new_tokens = 512  # Scale-RL: longer context for reasoning chains
temperature = 1.0  # Encourage exploration
top_k = 50  # Standard sampling

# Batch sizes
device_batch_size = 4  # Max forward pass size to avoid OOM
examples_per_step = 8  # Prompts per gradient step (across all ranks)
num_samples = 16  # Samples per prompt for variance reduction

# Learning rates (Scale-RL: use ~0.5x SFT rates for RL stability)
unembedding_lr = 0.002
embedding_lr = 0.1
matrix_lr = 0.01
weight_decay = 0.0

# LR schedule
init_lr_frac = 0.05  # Start at 5% of base LR
warmup_frac = 0.05  # 5% warmup

# Training
num_epochs = 2
save_every = 100
eval_every = 50
eval_examples = 200

# Scale-RL: Zero-variance prompt filtering
use_prompt_filtering = True
prompt_filter_min_samples = 8
prompt_filter_threshold = 0.95

# Scale-RL: Curriculum learning
use_curriculum = True
curriculum_frac = 0.3  # First 30% of training uses curriculum

# Parse CLI overrides
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open(os.path.join("nanochat", "configurator.py")).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Initialize compute and precision

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# Wandb logging
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat-cot-rl", name=run, config=user_config)
)

# -----------------------------------------------------------------------------
# Load model and tokenizer

model, tokenizer, meta = load_model(
    source, device, phase="eval", model_tag=model_tag, step=step
)
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Task setup

train_task = MathCoT(subset="main", split="train", cot_prompt=True)
val_task = MathCoT(subset="main", split="test", cot_prompt=True)

# Scale-RL: Prompt tracker for zero-variance filtering
prompt_tracker = (
    PromptTracker(
        min_samples=prompt_filter_min_samples,
        skip_threshold=prompt_filter_threshold,
    )
    if use_prompt_filtering
    else None
)

# Scale-RL: Curriculum scheduler
curriculum_scheduler = (
    CurriculumScheduler(
        prompt_tracker=prompt_tracker,
        curriculum_frac=curriculum_frac,
    )
    if use_curriculum and prompt_tracker
    else None
)

# Calculate training steps
num_steps = (len(train_task) // examples_per_step) * num_epochs
print0(f"Training task size: {len(train_task)}")
print0(f"Number of training steps: {num_steps}")
print0(f"Samples per prompt: {num_samples}")
print0(f"Max new tokens: {max_new_tokens}")

# -----------------------------------------------------------------------------
# Rollout generator


@torch.no_grad()
def get_batch(current_step):
    """
    Generate rollouts for one training step.

    Implements Scale-RL best practices:
    - Multiple samples per prompt
    - Zero-variance prompt filtering
    - Curriculum learning
    - Advantage normalization (r - μ)
    """
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = list(range(ddp_rank, len(train_task), ddp_world_size))

    for example_idx in itertools.cycle(rank_indices):
        # Get conversation
        conversation = train_task[example_idx]
        prompt_id = conversation.get("index", example_idx)

        # Scale-RL: Zero-variance filtering
        if prompt_tracker and prompt_tracker.should_skip(prompt_id):
            continue

        # Scale-RL: Curriculum learning
        if curriculum_scheduler and not curriculum_scheduler.should_include(
            prompt_id, current_step, num_steps
        ):
            continue

        # Tokenize for completion
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate samples in batches to avoid OOM
        model.eval()
        generated_sequences = []
        masks = []
        num_batches = num_samples // device_batch_size

        for batch_idx in range(num_batches):
            seed = hash((current_step, example_idx, batch_idx)) & 0x7FFFFFFF
            with autocast_ctx:
                seqs, batch_masks = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,
                )
            generated_sequences.extend(seqs)
            masks.extend(batch_masks)

        # Compute rewards for each sample
        rewards = []
        for seq in generated_sequences:
            generated_tokens = seq[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Update prompt tracker
        if prompt_tracker:
            prompt_tracker.update(prompt_id, rewards)

        # Pad sequences to same length
        max_length = max(len(seq) for seq in generated_sequences)
        padded_seqs = [
            seq + [assistant_end] * (max_length - len(seq))
            for seq in generated_sequences
        ]
        padded_masks = [m + [0] * (max_length - len(m)) for m in masks]

        # Convert to tensors
        ids = torch.tensor(padded_seqs, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)

        # Create autoregressive inputs/targets
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1  # Mask non-sampled tokens

        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        # Scale-RL: Advantage normalization - (r - μ) only, no division by σ
        mu = rewards.mean()
        advantages = rewards - mu

        yield generated_sequences, inputs, targets, rewards, advantages, prompt_id


# -----------------------------------------------------------------------------
# Evaluation


def run_eval(task, max_examples=None, num_samples_eval=1, temperature_eval=0.0):
    """
    Evaluate pass@k on the task.
    """
    max_examples = min(max_examples, len(task)) if max_examples else len(task)
    results = []

    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        seqs, _ = engine.generate_batch(
            tokens,
            num_samples=num_samples_eval,
            max_tokens=max_new_tokens,
            temperature=temperature_eval,
            top_k=top_k if temperature_eval > 0 else None,
        )

        outcomes = []
        for seq in seqs:
            generated_text = tokenizer.decode(seq[prefix_length:])
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append(is_correct)

        results.append(outcomes)

    return results


# -----------------------------------------------------------------------------
# Optimizer setup

optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)

# Set initial LR
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]


def get_lr_multiplier(step_idx):
    """LR schedule with warmup then linear decay."""
    warmup_steps = int(num_steps * warmup_frac)
    if step_idx < warmup_steps:
        return step_idx / warmup_steps
    return 1.0 - (step_idx - warmup_steps) / (num_steps - warmup_steps)


# -----------------------------------------------------------------------------
# Training loop

assert examples_per_step % ddp_world_size == 0
examples_per_rank = examples_per_step // ddp_world_size
print0(f"Examples per rank per step: {examples_per_rank}")
print0(f"Total sequences per step: {examples_per_step * num_samples}")

batch_iterator = None
for step in range(num_steps):

    # Lazily create batch iterator (needs current step for curriculum)
    if batch_iterator is None:
        batch_iterator = get_batch(step)

    # -----------------------------
    # Evaluation
    # -----------------------------
    if step % eval_every == 0:
        model.eval()

        with autocast_ctx:
            results = run_eval(
                val_task,
                max_examples=eval_examples,
                num_samples_eval=device_batch_size,
                temperature_eval=1.0,
            )

        # Compute pass@k
        passk = torch.zeros(device_batch_size, device=device)
        for k in range(1, device_batch_size + 1):
            passk[k - 1] = sum(any(r[:k]) for r in results if r)

        num_results = torch.tensor(len(results), device=device)
        if ddp:
            dist.all_reduce(num_results, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)

        passk = passk / num_results.item()

        # Log pass@k
        passk_strs = [
            f"Pass@{k}: {passk[k-1].item():.4f}" for k in [1, 4, device_batch_size]
        ]
        print0(f"Step {step} | {', '.join(passk_strs)}")

        wandb_run.log(
            {
                "step": step,
                **{
                    f"pass@{k}": passk[k - 1].item()
                    for k in range(1, device_batch_size + 1)
                },
            }
        )

        # Log prompt tracker stats
        if prompt_tracker:
            stats = prompt_tracker.get_stats_summary()
            print0(f"  Prompt tracker: {stats}")
            wandb_run.log(
                {"step": step, **{f"tracker/{k}": v for k, v in stats.items()}}
            )

    # -----------------------------
    # Forward/Backward
    # -----------------------------
    rewards_list = []
    seq_lengths = []

    for _ in range(examples_per_rank):
        # Get batch for one example
        seqs, inputs, targets, rewards, advantages, prompt_id = next(batch_iterator)

        model.train()

        # Process in sub-batches to avoid OOM
        num_passes = inputs.size(0) // device_batch_size

        for pass_idx in range(num_passes):
            b0 = pass_idx * device_batch_size
            b1 = (pass_idx + 1) * device_batch_size

            batch_inputs = inputs[b0:b1]
            batch_targets = targets[b0:b1]
            batch_advantages = advantages[b0:b1]

            # Forward pass
            # Scale-RL: FP32 for logits (done in model.forward via logits.float())
            with autocast_ctx:
                logp = -model(
                    batch_inputs, batch_targets, loss_reduction="none"
                ).view_as(batch_inputs)

            # Scale-RL: Token-level advantages (GAPO-style)
            pg_obj = (logp * batch_advantages.unsqueeze(-1)).sum()

            # Normalize by valid tokens and total passes
            num_valid = (batch_targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)

            # Loss = negative objective (we minimize loss, maximize objective)
            loss = -pg_obj
            loss.backward()

        rewards_list.append(rewards.mean().item())
        seq_lengths.extend(len(seq) for seq in seqs)

    # Aggregate logging stats
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_seq_len = sum(seq_lengths) / len(seq_lengths)

    if ddp:
        mean_reward_t = torch.tensor(mean_reward, device=device)
        mean_seq_len_t = torch.tensor(mean_seq_len, device=device)
        dist.all_reduce(mean_reward_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_seq_len_t, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_t.item()
        mean_seq_len = mean_seq_len_t.item()

    print0(
        f"Step {step}/{num_steps} | Reward: {mean_reward:.4f} | Seq len: {mean_seq_len:.1f}"
    )
    wandb_run.log(
        {
            "step": step,
            "reward": mean_reward,
            "seq_length": mean_seq_len,
        }
    )

    # -----------------------------
    # Optimizer step
    # -----------------------------
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    wandb_run.log({"step": step, "lrm": lrm})

    # Refresh batch iterator with new step (for curriculum)
    batch_iterator = get_batch(step + 1)

    # -----------------------------
    # Checkpointing
    # -----------------------------
    if master_process and (
        (step > 0 and step % save_every == 0) or step == num_steps - 1
    ):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_tag = model_tag if model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "cot_rl_checkpoints", output_tag)

        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,
            {"model_config": model.config.__dict__},
        )
        print(f"✅ Saved checkpoint to {checkpoint_dir}")

# -----------------------------------------------------------------------------
# Finalize

from nanochat.report import get_report

get_report().log(
    section="CoT RLVR",
    data=[
        user_config,
        {"final_reward": mean_reward, "num_steps": num_steps},
    ],
)

wandb_run.finish()
compute_cleanup()
