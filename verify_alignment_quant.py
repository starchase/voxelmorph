
import nibabel as nib
import numpy as np
from pathlib import Path
import random
from scipy.ndimage import center_of_mass

def get_body_mask(vol):
    # Simple thresholding to find body
    # CT usually has -1000 HU (air). Preprocessed might be 0-1 or different.
    # We'll use > mean as a rough approximation for foreground
    return vol > vol.mean()

def check_quant(ct_dir, mr_dir, num_samples=10):
    ct_files = sorted(list(Path(ct_dir).glob('*.nii.gz')))
    mr_files = sorted(list(Path(mr_dir).glob('*.nii.gz')))
    
    if not ct_files or not mr_files:
        print("No files found.")
        return

    print(f"Checking {num_samples} random unpaired samples for alignment...")
    dists = []
    
    for _ in range(num_samples):
        ct_p = random.choice(ct_files)
        mr_p = random.choice(mr_files)
        
        ct_vol = nib.load(str(ct_p)).get_fdata()
        mr_vol = nib.load(str(mr_p)).get_fdata()
        
        # Check dim
        if ct_vol.shape != mr_vol.shape:
            print(f"Shape mismatch! {ct_vol.shape} vs {mr_vol.shape}")
            continue
            
        # Calc center of mass
        com_ct = np.array(center_of_mass(get_body_mask(ct_vol)))
        com_mr = np.array(center_of_mass(get_body_mask(mr_vol)))
        
        dist = np.linalg.norm(com_ct - com_mr)
        dists.append(dist)
        print(f"CT: {ct_p.name}, MR: {mr_p.name} -> CoM Dist: {dist:.2f} voxels")
        
    avg_dist = np.mean(dists)
    print(f"\nAverage CoM Distance: {avg_dist:.2f} voxels")
    
    # Heuristic check
    limit = max(ct_vol.shape) * 0.15 # Allow 15% deviation
    if avg_dist < limit:
        print("CONCLUSION: Likely Affine Aligned (Centers are close).")
    else:
        print("CONCLUSION: Likely NOT Aligned (Centers represent large offset).")

if __name__ == "__main__":
    check_quant('processed/ct/train/image', 'processed/mr/train/image')
