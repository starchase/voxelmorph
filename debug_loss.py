
import torch
import neurite as ne
import neurite.nn.modules

def check_losses():
    print("Checking NCC behavior...")
    # Create two identical tensors
    t1 = torch.rand(1, 1, 32, 32)
    t2 = t1.clone()
    
    # Calculate NCC
    ncc_mod = ne.nn.modules.NCC()
    val_perfect = ncc_mod(t1, t2).item()
    print(f"NCC for identical images (should be ~1.0): {val_perfect}")
    
    # Calculate MSE
    mse_mod = ne.nn.modules.MSE()
    val_mse = mse_mod(t1, t2).item()
    print(f"MSE for identical images (should be 0.0): {val_mse}")

    # Create distinct tensors
    t3 = torch.rand(1, 1, 32, 32)
    val_diff = ncc_mod(t1, t3).item()
    print(f"NCC for random images (should be < 1.0): {val_diff}")

if __name__ == "__main__":
    check_losses()
