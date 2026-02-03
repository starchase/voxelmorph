
import nibabel as nib
import numpy as np
from pathlib import Path

def check_stats(directory):
    directory = Path(directory)
    if not directory.exists():
        return

    print(f"Checking {directory}...")
    files = list(directory.glob('*.nii.gz'))
    if not files:
        return

    # Check first 5 files
    for f in files[:5]:
        img = nib.load(str(f))
        data = img.get_fdata()
        print(f"  {f.name}: Shape {data.shape}, Range [{data.min():.2f}, {data.max():.2f}], Mean {data.mean():.2f}")

print("Checking Train CT:")
check_stats('processed/ct/train/image')
print("Checking Train MR:")
check_stats('processed/mr/train/image')
