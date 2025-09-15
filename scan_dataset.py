import numpy as np
import os
from glob import glob

def compute_flow_minmax(root_dir):
    print(f"Scanning dataset at: {root_dir}")
    min_val = float("inf")
    max_val = float("-inf")

    # Find all .npy files
    files = glob(os.path.join(root_dir, "**/*.npy"), recursive=True)
    print(f"Found {len(files)} files")

    for i, path in enumerate(files):
        arr = np.load(path, mmap_mode="r")  # mmap avoids loading all into RAM
        local_min = arr.min()
        local_max = arr.max()

        if local_min < min_val:
            min_val = local_min
        if local_max > max_val:
            max_val = local_max

        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(files)} files... "
                  f"Current min={min_val:.4f}, max={max_val:.4f}")

    print("\n===== Optical Flow Global Range =====")
    print(f"Global Min: {min_val:.4f}")
    print(f"Global Max: {max_val:.4f}")
    print("=====================================")

    return min_val, max_val

if __name__ == "__main__":
    dataset_root = "optical_flow_dataset"  # adjust if needed
    min_val, max_val = compute_flow_minmax(dataset_root)
