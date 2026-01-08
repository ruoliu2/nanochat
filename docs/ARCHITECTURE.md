# Architecture

## Training Pipeline

```mermaid
flowchart TB
    subgraph Data["Data Preparation"]
        D1[FineWeb-Edu] --> D2[Parquet Shards]
        D2 --> D3[Streaming Loader]
    end

    subgraph Tok["Tokenizer"]
        T1[RustBPE Train] --> T2[tiktoken Export]
    end

    subgraph Train["Training Stages"]
        B1[Pretrain] --> B2[Midtrain]
        B2 --> B3[SFT]
        B3 --> B4[RL]
    end

    subgraph Deploy["Deployment"]
        W1[Web Server]
        W2[CLI]
    end

    D3 --> B1
    T2 --> B1
    B4 --> W1
    B4 --> W2
```

## Model Architecture

```mermaid
flowchart TD
    IN[Input Tokens] --> EMB[Token Embedding]
    EMB --> N1[RMSNorm]
    N1 --> TB[Transformer Blocks x N]

    subgraph TB[Transformer Block]
        ATT[Group-Query Attention<br/>+ Rotary Embeddings<br/>+ QK Norm]
        MLP[MLP with ReLU²]
        ATT --> MLP
    end

    TB --> N2[RMSNorm]
    N2 --> OUT[Output Projection]
    OUT --> LOG[Logits]
```

## Key Design Choices

| Feature | Choice | Rationale |
|---------|--------|-----------|
| Position | Rotary embeddings | More efficient |
| Attention | Group-Query | Less KV cache memory |
| Activation | ReLU² | Better empirically |
| Norm | RMSNorm | Simpler, no params |
| Weights | Untied | Better performance |

## Optimization

```mermaid
flowchart LR
    subgraph Params
        M[Matrix 2D+] --> MU[Muon Optimizer]
        E[Embeddings 1D] --> AW[AdamW ZeRO-2]
    end
```

## Data Flow

```mermaid
flowchart LR
    P[Parquet] --> DL[DataLoader]
    DL --> |stream| BPE[Tokenizer]
    BPE --> |batch| GPU[GPU Training]
```

## Distributed Training

- ZeRO-2 sharded optimizer states
- DDP-aware data loading
- Gradient accumulation
- All-reduce synchronization
