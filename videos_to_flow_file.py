import os
import shutil
import random

# Original folders
source_folders = [
    "optical_flow_dataset/crash",
    "optical_flow_dataset/no_crash"
]

# Destination folder
dest_folder = "flow_logistic_data"

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Subfolders to collect from source
source_subfolders = ["val", "test"]

# Copy and split files
for folder in source_folders:
    folder_name = os.path.basename(folder)  # "crash" or "no_crash"
    
    # Collect all files from val and test
    all_files = []
    for subfolder in source_subfolders:
        src_path = os.path.join(folder, subfolder)
        if os.path.exists(src_path):
            all_files.extend([os.path.join(src_path, f) for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))])
    
    # Shuffle files
    random.shuffle(all_files)
    total = len(all_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": all_files[:train_end],
        "val": all_files[train_end:val_end],
        "test": all_files[val_end:]
    }

    # Copy files to destination
    for split_name, files in splits.items():
        dst_path = os.path.join(dest_folder, folder_name, split_name)
        os.makedirs(dst_path, exist_ok=True)
        for file_path in files:
            shutil.copy(file_path, dst_path)

print("Files copied and split into train/val/test successfully!")
