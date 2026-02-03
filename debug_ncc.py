
import torch
import neurite as ne
import neurite.nn.functional as nef
import numpy as np

def check_ncc():
    # Simulate data
    B, C, D, H, W = 1, 1, 160, 192, 224
    t1 = torch.rand(B, C, D, H, W).cuda() if torch.cuda.is_available() else torch.rand(B, C, D, H, W)
    t2 = torch.rand(B, C, D, H, W).cuda() if torch.cuda.is_available() else torch.rand(B, C, D, H, W)
    
    # Normalized to [0, 1]
    t1 = (t1 - t1.min()) / (t1.max() - t1.min())
    t2 = (t2 - t2.min()) / (t2.max() - t2.min())

    ncc_mod = ne.nn.modules.NCC(window_size=21)
    
    loss = ncc_mod(t1, t2)
    print(f"NCC value (mean): {loss.item()}")
    
    # Check bounds
    print(f"Is NCC in [0, 1]? {0 <= loss.item() <= 1}")
    
    # Try with identical
    loss_id = ncc_mod(t1, t1)
    print(f"NCC identical (should be ~1): {loss_id.item()}")

    # Try components
    # Use internal functional if possible to see raw values
    raw_ncc = nef.ncc(t1, t2, window_size=21, reduction=None)
    print(f"Raw NCC shape: {raw_ncc.shape}")
    print(f"Raw NCC max: {raw_ncc.max().item()}, min: {raw_ncc.min().item()}")

if __name__ == "__main__":
    try:
        check_ncc()
    except Exception as e:
        print(f"Error: {e}")
