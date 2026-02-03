
import nibabel as nib
import numpy as np
import glob
import os

def check_files(file_pattern, modality):
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"No files found for {file_pattern}")
        return

    print(f"--- Checking {modality} ({len(files)} files found) ---")
    
    not_normalized_count = 0
    not_clipped_count = 0 
    
    for i, f in enumerate(files):
        try:
            img = nib.load(f)
            data = img.get_fdata()
            dmin = data.min()
            dmax = data.max()
            
            # Check for strict [0, 1] normalization
            is_0_1 = (dmin >= 0) and (dmax <= 1.0)
            
            if not is_0_1:
                not_normalized_count += 1
                msg = f"[NOT NORMALIZED] {os.path.basename(f)}: Min={dmin:.2f}, Max={dmax:.2f}"
                
                if "CT" in modality:
                     # Check if it fits within [-1000, 1000] range (approx)
                     is_clipped = (dmin >= -1000) and (dmax <= 1000)
                     if not is_clipped:
                         not_clipped_count += 1
                         msg += " [NOT CLIPPED to -1000,1000]"
                
                print(msg)

        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Summary for {modality}:")
    print(f"  Total files: {len(files)}")
    print(f"  Not in [0, 1]: {not_normalized_count}")
    if "CT" in modality:
        print(f"  Not clipped to [-1000, 1000] (among non-normalized): {not_clipped_count}")

splits = ['train', 'val', 'test']
modalities = ['ct', 'mr']

for split in splits:
    for mod in modalities:
        # Path construction: processed/ct/train/image/*.nii.gz
        pattern = os.path.join('processed', mod, split, 'image', '*.nii.gz')
        print(f"\nChecking {split.upper()} {mod.upper()}:")
        check_files(pattern, f"{mod.upper()} ({split})")
