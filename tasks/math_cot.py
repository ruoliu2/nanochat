"""
Math tasks optimized for Chain-of-Thought RLVR training.

Combines GSM8K with CoT-friendly prompting and extended reward functions
that encourage step-by-step reasoning while maintaining verifiable rewards.
"""

import re
from datasets import load_dataset
from tasks.common import Task


# Regex to extract final answer after #### marker (same as GSM8K)
ANSWER_RE = re.compile(r"####\s*(\-?[0-9\.\,]+)")

# Regex to extract numbers from text (fallback)
NUMBER_RE = re.compile(r"(\-?[0-9]+(?:\.[0-9]+)?)")


def normalize_number(s):
    """Normalize a number string for comparison."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        # Convert to float then back to remove trailing zeros
        return str(float(s))
    except ValueError:
        return s


def extract_answer(text):
    """
    Extract the final numerical answer from text.
    Priority: #### marker > last number in text
    """
    # First try to find #### marker (preferred)
    match = ANSWER_RE.search(text)
    if match:
        return normalize_number(match.group(1))
    
    # Fallback: find the last number in the text
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return normalize_number(numbers[-1])
    
    return None


def count_reasoning_steps(text):
    """
    Count the number of reasoning steps in the text.
    Heuristic: count sentences that contain calculations or explanations.
    """
    # Split on newlines and periods
    parts = re.split(r'[\n\.]', text)
    steps = 0
    for part in parts:
        part = part.strip()
        if len(part) > 10:  # Non-trivial content
            # Contains numbers or calculation-like patterns
            if NUMBER_RE.search(part) or '=' in part or '+' in part or '-' in part or '*' in part or '/' in part:
                steps += 1
    return steps


class MathCoT(Task):
    """
    Math task with Chain-of-Thought prompting for RLVR.
    Based on GSM8K but with CoT-optimized prompts and reward shaping.
    """

    def __init__(self, subset="main", split="train", cot_prompt=True, **kwargs):
        """
        Args:
            subset: GSM8K subset ("main" or "socratic")
            split: "train" or "test"
            cot_prompt: Whether to add CoT system prompt
        """
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "subset must be main|socratic"
        assert split in ["train", "test"], "split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)
        self.cot_prompt = cot_prompt

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """Get a single problem from the dataset."""
        row = self.ds[index]
        question = row['question']
        answer = row['answer']
        
        # Add CoT-encouraging prompt to the question
        if self.cot_prompt:
            question = (
                "Solve this step by step. Show your work, then give the final "
                "numerical answer after ####.\n\n" + question
            )
        
        # Parse the reference answer which uses <<expr=result>> tool calls
        assistant_parts = self._parse_gsm8k_answer(answer)
        
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_parts},
        ]
        
        return {
            "messages": messages,
            "index": index,  # For prompt tracking
        }

    def _parse_gsm8k_answer(self, answer):
        """Parse GSM8K answer format with <<expr=result>> tool calls."""
        parts = []
        segments = re.split(r'(<<[^>]+>>)', answer)
        
        for segment in segments:
            if segment.startswith('<<') and segment.endswith('>>'):
                inner = segment[2:-2]
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                parts.append({"type": "python", "text": expr})
                parts.append({"type": "python_output", "text": result})
            else:
                parts.append({"type": "text", "text": segment})
        
        return parts

    def evaluate(self, conversation, assistant_response):
        """
        Evaluate if the response is correct.
        Returns 1 for correct, 0 for incorrect.
        """
        # Extract reference answer from conversation
        assistant_msg = conversation['messages'][-1]
        assert assistant_msg['role'] == "assistant"
        
        # Get the ground truth from the last text part
        if isinstance(assistant_msg['content'], list):
            last_text = assistant_msg['content'][-1]['text']
        else:
            last_text = assistant_msg['content']
        
        ref_answer = extract_answer(last_text)
        pred_answer = extract_answer(assistant_response)
        
        return int(ref_answer == pred_answer)

    def reward(self, conversation, assistant_response):
        """
        Compute reward for RLVR training.
        
        Base reward is correctness (0 or 1).
        Optional bonuses for:
        - Using #### format correctly
        - Having reasoning steps
        """
        # Base reward: correctness
        is_correct = self.evaluate(conversation, assistant_response)
        reward = float(is_correct)
        
        # Format bonus: using #### marker (small bonus)
        if "####" in assistant_response:
            reward += 0.02
        
        # Reasoning bonus: having multiple steps (very small bonus)
        # Only give if answer is correct to not reward verbose wrong answers
        if is_correct:
            num_steps = count_reasoning_steps(assistant_response)
            if num_steps >= 2:
                reward += 0.01
        
        # Cap reward at 1.05 (correctness dominates)
        return min(reward, 1.05)

    def get_prompt_id(self, conversation):
        """Get a unique ID for prompt tracking."""
        return conversation.get("index", hash(str(conversation['messages'][0])))


class MathCoTMix(Task):
    """
    Mixture of math tasks for curriculum learning.
    Starts with simpler problems, gradually includes harder ones.
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        
        # Load both main and socratic (socratic has step-by-step hints)
        self.main_ds = load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=42)
        
        # Estimate difficulty by answer length (longer answers = harder)
        self.problems = []
        for i, row in enumerate(self.main_ds):
            difficulty = len(row['answer'])  # Simple heuristic
            self.problems.append({
                "index": i,
                "difficulty": difficulty,
                "row": row,
            })
        
        # Sort by difficulty for curriculum
        self.problems.sort(key=lambda x: x['difficulty'])

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.problems)

    def get_example(self, index):
        """Get problem by curriculum-sorted index."""
        problem = self.problems[index]
        row = problem['row']
        
        question = (
            "Solve this step by step. Show your work, then give the final "
            "numerical answer after ####.\n\n" + row['question']
        )
        
        # Parse answer
        parts = []
        segments = re.split(r'(<<[^>]+>>)', row['answer'])
        for segment in segments:
            if segment.startswith('<<') and segment.endswith('>>'):
                inner = segment[2:-2]
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                parts.append({"type": "python", "text": expr})
                parts.append({"type": "python_output", "text": result})
            else:
                parts.append({"type": "text", "text": segment})
        
        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": parts},
            ],
            "index": problem['index'],
            "difficulty": problem['difficulty'],
        }

    def evaluate(self, conversation, assistant_response):
        """Evaluate correctness."""
        assistant_msg = conversation['messages'][-1]
        if isinstance(assistant_msg['content'], list):
            last_text = assistant_msg['content'][-1]['text']
        else:
            last_text = assistant_msg['content']
        
        ref_answer = extract_answer(last_text)
        pred_answer = extract_answer(assistant_response)
        return int(ref_answer == pred_answer)

    def reward(self, conversation, assistant_response):
        """Compute RLVR reward."""
        is_correct = self.evaluate(conversation, assistant_response)
        reward = float(is_correct)
        
        if "####" in assistant_response:
            reward += 0.02
        
        if is_correct:
            num_steps = count_reasoning_steps(assistant_response)
            if num_steps >= 2:
                reward += 0.01
        
        return min(reward, 1.05)


if __name__ == "__main__":
    # Quick test
    task = MathCoT(split="train")
    print(f"Number of examples: {len(task)}")
    
    ex = task[0]
    print("\n=== Example 0 ===")
    print("User:", ex['messages'][0]['content'][:200], "...")
    
    # Test reward function
    test_response = "Let me solve this step by step.\n2 + 2 = 4\n#### 4"
    reward = task.reward(ex, test_response)
    print(f"\nTest reward: {reward}")
    
    # Test curriculum mix
    mix = MathCoTMix(split="train")
    print(f"\nCurriculum mix size: {len(mix)}")
    print(f"Easiest problem difficulty: {mix.problems[0]['difficulty']}")
    print(f"Hardest problem difficulty: {mix.problems[-1]['difficulty']}")
