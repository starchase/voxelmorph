import os
import glob
import numpy as np
import nibabel as nib

def analyze_folder(folder_path, name):
    print(f"\n--- Analyzing {name} in {folder_path} ---")
    files = sorted(glob.glob(os.path.join(folder_path, "*.nii.gz")))
    if not files:
        print("No .nii.gz files found.")
        return

    # Check a few files
    sample_files = files[:3]
    
    overall_min = float('inf')
    overall_max = float('-inf')

    for f in sample_files:
        print(f"Loading {os.path.basename(f)}...")
        img = nib.load(f)
        data = img.get_fdata()
        
        dmin = np.min(data)
        dmax = np.max(data)
        dmean = np.mean(data)
        dstd = np.std(data)
        
        overall_min = min(overall_min, dmin)
        overall_max = max(overall_max, dmax)

        print(f"  Shape: {data.shape}")
        print(f"  Range: [{dmin:.4f}, {dmax:.4f}]")
        print(f"  Mean: {dmean:.4f}, Std: {dstd:.4f}")
        
        # Checking for common HU values or normalization
        if name == "CT":
            # Check for air (-1000) or bone (>1000)
            print(f"  Percentiles (0, 1, 5, 50, 95, 99, 100): {np.percentile(data, [0, 1, 5, 50, 95, 99, 100])}")
        else: # MR
             print(f"  Percentiles (0, 1, 5, 50, 95, 99, 100): {np.percentile(data, [0, 1, 5, 50, 95, 99, 100])}")

    print(f"\nSummary for {name}:")
    print(f"  Overall Min: {overall_min}")
    print(f"  Overall Max: {overall_max}")
    
    if overall_min >= 0 and overall_max <= 1.0:
        print(f"  -> Seems to be Min-Max Normalized to [0, 1]")
    elif overall_min >= -1024 and overall_max > 1000:
        print(f"  -> Seems to be raw HU or wide window")
    else:
        print(f"  -> Custom range.")

ct_path = "/root/voxelmorph-dev/processed/ct/train_pair/image"
mr_path = "/root/voxelmorph-dev/processed/mr/train_pair/image"

analyze_folder(ct_path, "CT")
analyze_folder(mr_path, "MR")
