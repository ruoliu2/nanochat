# scripts/ - Training & Evaluation

Scripts for the complete training pipeline.

## Pipeline Flow

```mermaid
flowchart LR
    subgraph Tokenizer
        TT[tok_train.py] --> TE[tok_eval.py]
    end

    subgraph Base
        BT[base_train.py] --> BL[base_loss.py]
        BL --> BE[base_eval.py]
    end

    subgraph Chat
        MT[mid_train.py] --> SF[chat_sft.py]
        SF --> RL[chat_rl.py]
        RL --> CE[chat_eval.py]
    end

    subgraph Deploy
        CLI[chat_cli.py]
        WEB[chat_web.py]
    end

    TE --> BT
    BE --> MT
    CE --> CLI
    CE --> WEB
```

## Scripts by Stage

### Tokenizer
| Script | Purpose |
|--------|---------|
| `tok_train.py` | Train BPE tokenizer |
| `tok_eval.py` | Evaluate compression |

### Base Model
| Script | Purpose |
|--------|---------|
| `base_train.py` | Pretrain with Chinchilla scaling |
| `base_loss.py` | Evaluate validation BPB |
| `base_eval.py` | CORE benchmark |

### Chat Model
| Script | Purpose |
|--------|---------|
| `mid_train.py` | Teach conversation tokens |
| `chat_sft.py` | Supervised fine-tuning |
| `chat_rl.py` | Reinforcement learning |
| `chat_eval.py` | Evaluate chat model |

### Deployment
| Script | Purpose |
|--------|---------|
| `chat_cli.py` | CLI chat interface |
| `chat_web.py` | FastAPI web server |

## Usage

```bash
# Pretrain (8 GPUs)
torchrun --nproc_per_node=8 scripts/base_train.py

# Web server
python scripts/chat_web.py --checkpoint path/to/model
```
