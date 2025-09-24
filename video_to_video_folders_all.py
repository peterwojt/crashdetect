import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
VIDEOS_DIR = "videos"
LABELS_CSV = "labels.csv"
OUTPUT_DIR = "car_crash_video_dataset2"

# Create output directories
splits = ['train', 'val', 'test']
classes = ['crash', 'non_crash']
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# Read labels
df = pd.read_csv(LABELS_CSV)

# Keep only labels 'c', 'n', 'z'
df = df[df['label'].isin(['c', 'n', 'z'])]

# Map labels to classes
df['class'] = df['label'].apply(lambda x: 'crash' if x == 'c' else 'non_crash')

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Stratified split
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['class'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['class'])

split_map = {
    'train': train_df,
    'val': val_df,
    'test': test_df
}

# Copy files into the output directory
for split, split_df in split_map.items():
    for _, row in split_df.iterrows():
        src = os.path.join(VIDEOS_DIR, row['filename'])
        dst = os.path.join(OUTPUT_DIR, split, row['class'], row['filename'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found.")

print("Done.")
