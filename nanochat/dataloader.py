from collections import deque
from dataclasses import dataclass

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer


@dataclass
class TokenWithBoundary:
    """A token with metadata about whether it starts a new document."""

    token_id: int
    is_doc_start: bool  # True if this token is the first token of a new document


def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
            first_pass = False
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


def packing_dataloader_with_state(
    B,
    T,
    split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
):
    """
    Sequence packing dataloader that tracks document boundaries.

    Returns (inputs, targets, loss_mask, state_dict) where:
    - inputs: (B, T) token ids
    - targets: (B, T) next token ids
    - loss_mask: (B, T) mask where 0 = don't compute loss (document boundary)
    - state_dict: for checkpointing/resume

    The loss_mask marks positions where the target token is from a DIFFERENT document
    than the input token. These positions should be ignored in loss computation because
    predicting the first token of a new document from the last token of the previous
    document is essentially random noise.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    def document_batches():
        parquet_paths = list_parquet_files()
        assert (
            len(parquet_paths) != 0
        ), "No dataset parquet files found, did you run dataset.py?"
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = (
            resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        )
        resume_rg_idx = (
            resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        )
        first_pass = True
        pq_idx = resume_pq_idx
        while True:
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths):
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                if (
                    first_pass
                    and (resume_rg_idx is not None)
                    and (pq_idx == resume_pq_idx)
                ):
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column("text").to_pylist()
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i : i + tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size
                pq_idx += 1
            first_pass = False

    batches = document_batches()

    # Now emit batches of tokens with boundary tracking
    needed_tokens = B * T + 1  # +1 for the final target
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # Buffer holds (token_id, is_doc_start) tuples
    token_buffer = deque()  # deque of (token_id, is_doc_start)

    while True:
        # Accumulate enough tokens
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(
                doc_batch, prepend=bos_token, num_threads=tokenizer_threads
            )
            for tokens in token_lists:
                # First token of each document is marked as doc_start
                for i, tok in enumerate(tokens):
                    token_buffer.append((tok, i == 0))

        # Extract tokens and boundary info
        tokens = []
        is_doc_starts = []
        for _ in range(needed_tokens):
            tok, is_start = token_buffer.popleft()
            tokens.append(tok)
            is_doc_starts.append(is_start)

        # Build tensors
        use_cuda_optimizations = device == "cuda"
        tokens_tensor = torch.tensor(
            tokens, dtype=torch.long, pin_memory=use_cuda_optimizations
        )

        # inputs[i] predicts targets[i], where targets[i] = tokens[i+1]
        inputs_cpu = tokens_tensor[:-1]
        targets_cpu = tokens_tensor[1:]

        # loss_mask[i] = 0 if targets[i] is the start of a new document
        # (i.e., is_doc_starts[i+1] == True means position i should be masked)
        is_doc_starts_tensor = torch.tensor(
            is_doc_starts[1:], dtype=torch.bool
        )  # skip first, align with targets
        loss_mask_cpu = (~is_doc_starts_tensor).to(
            torch.long
        )  # 1 = compute loss, 0 = ignore

        # Reshape and move to device
        inputs = inputs_cpu.view(B, T).to(
            device=device, non_blocking=use_cuda_optimizations
        )
        targets = targets_cpu.view(B, T).to(
            device=device, non_blocking=use_cuda_optimizations
        )
        loss_mask = loss_mask_cpu.view(B, T).to(
            device=device, non_blocking=use_cuda_optimizations
        )

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}
        yield inputs, targets, loss_mask, state_dict


def packing_dataloader(*args, **kwargs):
    """Helper that only emits inputs, targets, loss_mask (no state_dict)."""
    for inputs, targets, loss_mask, _state_dict in packing_dataloader_with_state(
        *args, **kwargs
    ):
        yield inputs, targets, loss_mask
