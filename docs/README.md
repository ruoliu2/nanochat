# nanochat Documentation

A complete ChatGPT implementation in a minimal, hackable codebase. Trains end-to-end on a single 8xH100 node for ~$100.

## Training Pipeline

```mermaid
flowchart LR
    A[Data] --> B[Tokenizer]
    B --> C[Pretrain]
    C --> D[Midtrain]
    D --> E[SFT]
    E --> F[Deploy]
```

## Quick Links

| Document | Description |
|----------|-------------|
| [STRUCTURE.md](./STRUCTURE.md) | Repository layout and file organization |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System architecture and component design |
| [OPTIMIZATIONS.md](./OPTIMIZATIONS.md) | Performance optimizations (FP8, sequence packing) |

## Module Documentation

| Directory | Description |
|-----------|-------------|
| [nanochat/](./nanochat/README.md) | Core Python package (model, tokenizer, engine) |
| [scripts/](./scripts/README.md) | Training and evaluation scripts |
| [tasks/](./tasks/README.md) | Evaluation task implementations |
| [rustbpe/](./rustbpe/README.md) | Rust BPE tokenizer |
| [dev/](./dev/README.md) | Development utilities |
| [tests/](./tests/README.md) | Unit tests |

## Project Goals

- **Minimal**: ~8,300 lines in 45 files
- **Hackable**: Single codebase, not a framework
- **Accessible**: <$1000 training budget

## Training Tiers

| Tier | Cost | Time | Model | Tokens |
|------|------|------|-------|--------|
| speedrun.sh | ~$100 | 4h | 561M params | 11.2B |
| run1000.sh | ~$800 | 33h | 1.9B params | 38B |
