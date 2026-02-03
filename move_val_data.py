import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

base_dir = '/root/voxelmorph-dev/classedAbdomenMRCT_norm'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

subdirs = ['images', 'labels', 'masks']
types = ['ct', 'mr']

# Create val directories
for subdir in subdirs:
    for t in types:
        path = os.path.join(val_dir, subdir, t)
        os.makedirs(path, exist_ok=True)

# Select and move files
for t in types:
    src_images_dir = os.path.join(train_dir, 'images', t)
    files = sorted(os.listdir(src_images_dir))
    
    # Filter only files (just in case)
    files = [f for f in files if os.path.isfile(os.path.join(src_images_dir, f))]
    
    # Randomly select 5
    selected_files = random.sample(files, 5)
    
    print(f"Selected {t} files: {selected_files}")
    
    for filename in selected_files:
        for subdir in subdirs:
            src_path = os.path.join(train_dir, subdir, t, filename)
            dst_path = os.path.join(val_dir, subdir, t, filename)
            
            if os.path.exists(src_path):
                print(f"Moving {src_path} to {dst_path}")
                shutil.move(src_path, dst_path)
            else:
                print(f"Warning: {src_path} does not exist!")

print("Transfer completed.")
