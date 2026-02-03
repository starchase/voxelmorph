
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path
import random

def check_alignment(ct_dir, mr_dir, num_samples=3):
    ct_files = sorted(list(Path(ct_dir).glob('*.nii.gz')))
    mr_files = sorted(list(Path(mr_dir).glob('*.nii.gz')))

    if not ct_files or not mr_files:
        print("Error: Could not find .nii.gz files in the specified directories.")
        return

    print(f"Found {len(ct_files)} CTs and {len(mr_files)} MRs.")
    
    # Randomly pick pairs that are LIKELY different patients (unpaired check)
    # Just picking random indices is sufficient
    for i in range(num_samples):
        ct_path = random.choice(ct_files)
        mr_path = random.choice(mr_files)
        
        print(f"Checking overlap: \n  CT: {ct_path.name}\n  MR: {mr_path.name}")
        
        ct_vol = nib.load(str(ct_path)).get_fdata()
        mr_vol = nib.load(str(mr_path)).get_fdata()
        
        # Take middle slice
        mid_z = ct_vol.shape[2] // 2
        ct_slice = ct_vol[:, :, mid_z]
        mr_slice = mr_vol[:, :, mid_z] # Assuming same Z depth

        # Normalize for display
        ct_slice = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
        mr_slice = (mr_slice - mr_slice.min()) / (mr_slice.max() - mr_slice.min())

        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(ct_slice, cmap='gray')
        plt.title('Random CT')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mr_slice, cmap='gray')
        plt.title('Random MR')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Overlay: Red channel = CT, Green channel = MR
        overlay = np.zeros((*ct_slice.shape, 3))
        overlay[..., 0] = ct_slice 
        overlay[..., 1] = mr_slice
        plt.imshow(overlay)
        plt.title('Overlay (Red=CT, Green=MR)')
        plt.axis('off')
        
        out_name = f'alignment_check_{i}.png'
        plt.savefig(out_name)
        print(f"Saved visualization to {out_name}")
        plt.close()

if __name__ == "__main__":
    # Adjust paths to where your data actually resides
    check_alignment(
        ct_dir='processed/ct/train/image', 
        mr_dir='processed/mr/train/image'
    )
