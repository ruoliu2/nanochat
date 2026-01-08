"""
Zero-variance prompt tracking for RLVR.

Implements one of the key Scale-RL best practices: skip prompts that
the model consistently gets right (or wrong), focusing compute on
prompts where there's actual learning signal.
"""

from collections import defaultdict


class PromptTracker:
    """
    Track which prompts have zero variance (always right or always wrong).
    This is a key Scale-RL insight: prompts the model always gets right
    or always gets wrong provide no learning signal and waste compute.
    """

    def __init__(self, min_samples=8, skip_threshold=0.95, decay=0.99):
        """
        Args:
            min_samples: Minimum samples before we consider skipping
            skip_threshold: Skip if success rate > threshold or < (1-threshold)
            decay: Exponential decay for old samples (keeps stats fresh)
        """
        self.min_samples = min_samples
        self.skip_threshold = skip_threshold
        self.decay = decay
        # prompt_id -> {"correct": float, "total": float}
        # Using floats for exponential decay
        self.stats = defaultdict(lambda: {"correct": 0.0, "total": 0.0})

    def update(self, prompt_id, rewards):
        """
        Update stats for a prompt given the rewards for samples.

        Args:
            prompt_id: Unique identifier for the prompt (e.g., dataset index)
            rewards: List of rewards (0 or 1) for each sample
        """
        entry = self.stats[prompt_id]
        # Apply decay to existing stats
        entry["correct"] *= self.decay
        entry["total"] *= self.decay
        # Add new samples
        entry["correct"] += sum(r > 0.5 for r in rewards)
        entry["total"] += len(rewards)

    def should_skip(self, prompt_id):
        """
        Check if we should skip this prompt due to zero variance.

        Returns:
            bool: True if prompt should be skipped
        """
        if prompt_id not in self.stats:
            return False

        entry = self.stats[prompt_id]
        if entry["total"] < self.min_samples:
            return False

        success_rate = entry["correct"] / entry["total"]
        # Skip if too easy (always right) or too hard (always wrong)
        return success_rate > self.skip_threshold or success_rate < (
            1 - self.skip_threshold
        )

    def get_success_rate(self, prompt_id):
        """Get the success rate for a prompt (for logging/debugging)."""
        if prompt_id not in self.stats:
            return None
        entry = self.stats[prompt_id]
        if entry["total"] < 1:
            return None
        return entry["correct"] / entry["total"]

    def get_stats_summary(self):
        """Get summary statistics for logging."""
        if not self.stats:
            return {"num_prompts": 0, "num_skippable": 0, "avg_success_rate": 0.0}

        num_prompts = len(self.stats)
        num_skippable = sum(1 for pid in self.stats if self.should_skip(pid))

        success_rates = []
        for entry in self.stats.values():
            if entry["total"] >= 1:
                success_rates.append(entry["correct"] / entry["total"])

        avg_success_rate = (
            sum(success_rates) / len(success_rates) if success_rates else 0.0
        )

        return {
            "num_prompts": num_prompts,
            "num_skippable": num_skippable,
            "avg_success_rate": avg_success_rate,
        }


class CurriculumScheduler:
    """
    Simple curriculum learning: start with easier problems.
    Implements another Scale-RL insight: curriculum helps small models.
    """

    def __init__(self, prompt_tracker, curriculum_frac=0.3):
        """
        Args:
            prompt_tracker: PromptTracker instance
            curriculum_frac: Fraction of training to apply curriculum
        """
        self.prompt_tracker = prompt_tracker
        self.curriculum_frac = curriculum_frac

    def should_include(self, prompt_id, current_step, total_steps):
        """
        Decide if a prompt should be included given current training progress.

        During early training (curriculum phase), prefer easier prompts.
        After curriculum phase, include all prompts.
        """
        progress = current_step / total_steps

        # After curriculum phase, include everything (except zero-variance)
        if progress > self.curriculum_frac:
            return not self.prompt_tracker.should_skip(prompt_id)

        # During curriculum phase, also skip hard prompts
        success_rate = self.prompt_tracker.get_success_rate(prompt_id)
        if success_rate is None:
            return True  # Haven't seen this prompt, include it

        # Gradually include harder prompts as training progresses
        # At start: only include prompts with >50% success
        # At end of curriculum: include prompts with >20% success
        min_success = 0.5 - (0.3 * progress / self.curriculum_frac)
        return success_rate >= min_success
