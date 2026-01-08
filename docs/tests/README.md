# tests/ - Unit Tests

Test suite for core components.

## Test Coverage

```mermaid
flowchart LR
    TR[test_rustbpe.py] --> |tests| TOK[Tokenizer]
    TE[test_engine.py] --> |tests| ENG[Engine]

    TOK --> |encoding| UC[Unicode]
    TOK --> |special| SP[Special tokens]

    ENG --> |generation| GEN[Token gen]
    ENG --> |tools| CALC[Calculator]
```

## Test Files

| File | Coverage |
|------|----------|
| `test_rustbpe.py` | Tokenizer edge cases, encoding/decoding |
| `test_engine.py` | Generation, KV cache, tools |

## Running

```bash
# All tests
pytest tests/

# Specific file
pytest tests/test_rustbpe.py -v
```
