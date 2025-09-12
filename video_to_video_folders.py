import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
VIDEOS_DIR = "videos"
LABELS_CSV = "labels.csv"
OUTPUT_DIR = "car_crash_video_dataset"

# Ratio of non-crash to crash videos
NON_CRASH_RATIO = 2  # Change this value as needed

# Create output directories
splits = ['train', 'val', 'test']
classes = ['crash', 'non_crash']
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# Read labels
df = pd.read_csv(LABELS_CSV)
df['class'] = df['label'].apply(lambda x: 'crash' if x == 'c' else 'non_crash')

# Select all crash videos
crash_df = df[df['class'] == 'crash']

# Sample non-crash videos
non_crash_df = df[df['class'] == 'non_crash']
n_crash = len(crash_df)
n_non_crash = min(len(non_crash_df), NON_CRASH_RATIO * n_crash)
non_crash_sampled_df = non_crash_df.sample(n=n_non_crash, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([crash_df, non_crash_sampled_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split data
train_df, temp_df = train_test_split(balanced_df, test_size=0.3, random_state=42, stratify=balanced_df['class'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['class'])

split_map = {
    'train': train_df,
    'val': val_df,
    'test': test_df
}

# Move files
for split, split_df in split_map.items():
    for _, row in split_df.iterrows():
        src = os.path.join(VIDEOS_DIR, row['filename'])
        dst = os.path.join(OUTPUT_DIR, split, row['class'], row['filename'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found.")

print("Done.")