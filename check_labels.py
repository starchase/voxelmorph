import nibabel as nib
import numpy as np

f = 'processed/ct/val/label/AbdomenMRCT_1012_0001.nii.gz'
img = nib.load(f)
data = img.get_fdata()
print(f"File: {f}")
print(f"Unique values: {np.unique(data)}")
print(f"Shape: {data.shape}")

f2 = 'processed/mr/val/label/AbdomenMRCT_1012_0000.nii.gz'
# Need to check if this file exists or which one matches 1012
import os
if os.path.exists(f2):
    img2 = nib.load(f2)
    data2 = img2.get_fdata()
    print(f"File: {f2}")
    print(f"Unique values: {np.unique(data2)}")
else:
    print(f"{f2} not found")
