# dev/ - Development Utilities

Helper scripts for development and experimentation.

## Files

```mermaid
flowchart TD
    GEN[gen_synthetic_data.py] --> |identity data| MID[Midtraining]
    REP[repackage_data_reference.py] --> |shows| DATA[Data prep]
    RUN[runcpu.sh] --> |example| CPU[CPU/MPS training]
```

| File | Purpose |
|------|---------|
| `gen_synthetic_data.py` | Generate identity conversations |
| `repackage_data_reference.py` | Data preparation reference |
| `runcpu.sh` | CPU/MPS training example |

## Synthetic Data

Customize model personality by mixing synthetic identity data:

```python
# Output format
{
    "messages": [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I am nanochat..."}
    ]
}
```

## CPU/MPS Training

`runcpu.sh` shows how to train smaller models on:
- Apple Silicon (MPS)
- CPU-only systems
- Memory-limited hardware
