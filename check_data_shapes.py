
import os
import nibabel as nib
import numpy as np
from pathlib import Path

def check_shapes(directory):
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory {directory} does not exist.")
        return

    shapes = {}
    print(f"Checking {directory}...")
    files = list(directory.glob('*.nii.gz'))
    if not files:
        print("No .nii.gz files found.")
        return

    for f in files:
        img = nib.load(str(f))
        shape = img.shape
        if shape not in shapes:
            shapes[shape] = 0
        shapes[shape] += 1
    
    for s, c in shapes.items():
        print(f"  Shape {s}: {c} files")

print("Checking Train Data:")
check_shapes('processed/ct/train/image')
check_shapes('processed/mr/train/image')

print("\nChecking Val Data:")
check_shapes('processed/ct/val/image')
check_shapes('processed/mr/val/image')
