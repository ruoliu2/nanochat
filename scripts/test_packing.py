"""
Quick sanity check for sequence packing implementation.
Run with: python -m scripts.test_packing

This script verifies:
1. The packing dataloader returns correct shapes
2. The loss_mask correctly identifies document boundaries
3. The model forward pass works with loss_mask
4. Loss values are reasonable (not NaN/Inf)
"""

import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_with_state,
    packing_dataloader_with_state,
)
from nanochat.tokenizer import get_tokenizer
from nanochat.common import print0

def test_packing():
    print0("=" * 60)
    print0("Testing Sequence Packing Implementation")
    print0("=" * 60)
    
    # Test parameters
    B, T = 2, 128  # small batch and sequence length
    device = "cpu"
    
    # Check if tokenizer exists
    try:
        tokenizer = get_tokenizer()
        print0(f"✓ Tokenizer loaded (vocab_size={tokenizer.get_vocab_size()})")
    except Exception as e:
        print0(f"✗ Tokenizer not found: {e}")
        print0("  Run: python -m scripts.tok_train --max_chars=10000000 --vocab_size=8192")
        return False
    
    # Create a tiny model
    print0("\n--- Creating tiny model ---")
    config = GPTConfig(
        sequence_len=T,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
    )
    model = GPT(config)
    model.init_weights()
    model.eval()
    print0(f"✓ Model created ({sum(p.numel() for p in model.parameters()):,} params)")
    
    # Test standard dataloader
    print0("\n--- Testing standard dataloader ---")
    try:
        std_loader = tokenizing_distributed_data_loader_with_state(
            B, T, split="train", device=device
        )
        x_std, y_std, state = next(std_loader)
        print0(f"✓ Standard loader: x={x_std.shape}, y={y_std.shape}")
        
        with torch.no_grad():
            loss_std = model(x_std, y_std)
        print0(f"✓ Standard loss: {loss_std.item():.4f}")
    except Exception as e:
        print0(f"✗ Standard dataloader failed: {e}")
        print0("  Make sure you have data: python -m nanochat.dataset -n 1")
        return False
    
    # Test packing dataloader
    print0("\n--- Testing packing dataloader ---")
    try:
        pack_loader = packing_dataloader_with_state(
            B, T, split="train", device=device
        )
        x_pack, y_pack, loss_mask, state = next(pack_loader)
        print0(f"✓ Packing loader: x={x_pack.shape}, y={y_pack.shape}, mask={loss_mask.shape}")
        
        # Check loss_mask statistics
        num_masked = (loss_mask == 0).sum().item()
        num_total = loss_mask.numel()
        pct_masked = 100 * num_masked / num_total
        print0(f"✓ Loss mask: {num_masked}/{num_total} positions masked ({pct_masked:.1f}%)")
        
        # Verify mask makes sense (should have SOME masked positions for document boundaries)
        if num_masked == 0:
            print0("  ⚠ Warning: No positions masked - this might be OK if documents are very long")
        elif num_masked == num_total:
            print0("  ✗ Error: ALL positions masked - something is wrong!")
            return False
        
        with torch.no_grad():
            loss_pack = model(x_pack, y_pack, loss_mask=loss_mask)
        print0(f"✓ Packing loss: {loss_pack.item():.4f}")
        
        # Check loss is valid
        if torch.isnan(loss_pack) or torch.isinf(loss_pack):
            print0("✗ Loss is NaN or Inf!")
            return False
            
    except Exception as e:
        print0(f"✗ Packing dataloader failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test that loss_mask actually affects loss
    print0("\n--- Verifying loss_mask effect ---")
    with torch.no_grad():
        # Loss without mask
        loss_no_mask = model(x_pack, y_pack, loss_mask=None)
        # Loss with mask
        loss_with_mask = model(x_pack, y_pack, loss_mask=loss_mask)
        
    print0(f"  Loss without mask: {loss_no_mask.item():.4f}")
    print0(f"  Loss with mask:    {loss_with_mask.item():.4f}")
    
    if num_masked > 0:
        if abs(loss_no_mask.item() - loss_with_mask.item()) < 1e-6:
            print0("  ⚠ Warning: Losses are identical - mask may not be working")
        else:
            print0("  ✓ Losses differ as expected (mask is working)")
    
    # Test backward pass
    print0("\n--- Testing backward pass ---")
    model.train()
    model.zero_grad()
    loss = model(x_pack, y_pack, loss_mask=loss_mask)
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print0(f"✓ Backward pass completed (grad_norm={grad_norm:.4f})")
    
    if grad_norm == 0:
        print0("  ✗ Warning: Gradient norm is zero!")
        return False
    
    print0("\n" + "=" * 60)
    print0("All tests passed! ✓")
    print0("=" * 60)
    print0("\nYou can now run full training with:")
    print0("  python -m scripts.base_train --use_packing=True ...")
    
    return True


if __name__ == "__main__":
    success = test_packing()
    exit(0 if success else 1)
