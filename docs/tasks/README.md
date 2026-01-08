# tasks/ - Evaluation Tasks

Task implementations for training and evaluation.

## Task Hierarchy

```mermaid
flowchart TD
    BASE[Task Base Class] --> MMLU[mmlu.py]
    BASE --> ARC[arc.py]
    BASE --> GSM[gsm8k.py]
    BASE --> HE[humaneval.py]
    BASE --> SB[spellingbee.py]
    BASE --> SM[smoltalk.py]
    BASE --> CJ[customjson.py]

    MIX[TaskMixture] --> |combines| MMLU
    MIX --> |combines| ARC
    SEQ[TaskSequence] --> |chains| GSM
```

## Available Tasks

| Task | File | Type |
|------|------|------|
| MMLU | `mmlu.py` | 57-subject MCQ |
| ARC | `arc.py` | Reasoning MCQ |
| GSM8K | `gsm8k.py` | Math generation |
| HumanEval | `humaneval.py` | Code generation |
| SpellingBee | `spellingbee.py` | Counting |
| SmolTalk | `smoltalk.py` | Conversation |
| CustomJSON | `customjson.py` | User-defined |

## Task Utilities

| Class | Purpose |
|-------|---------|
| `Task` | Base class with slicing |
| `TaskMixture` | Mix tasks with oversampling |
| `TaskSequence` | Sequential training |
| `render_mc()` | Format MCQ |

## Usage

```python
from tasks.mmlu import MMLU
from tasks.common import TaskMixture

mixture = TaskMixture([
    (MMLU(), 1.0),
    (ARC(), 2.0),  # 2x oversample
])
```
