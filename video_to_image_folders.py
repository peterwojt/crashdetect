import os
import cv2
import pandas as pd
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# -------- CONFIG --------
videos_dir = "videos"          # folder with videos
csv_file = "labels.csv"        # csv with columns: filename,label
output_dir = "data"            # root data folder
valid_labels = {"c", "z", "n"} # which labels we care about
max_size = 224                 # maximum width or height
train_ratio, val_ratio = 0.7, 0.15  # test gets the remainder
# -------------------------

# Prepare output dirs
splits = ["train", "val", "test"]
for split in splits:
    for label in valid_labels:
        Path(output_dir, split, label).mkdir(parents=True, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_file)
df = df[df["label"].isin(valid_labels)]

# Split dataset by video (not by frames!)
train_df, temp_df = train_test_split(df, test_size=(1-train_ratio), random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=(test_ratio := 1 - val_ratio/(1-train_ratio)), random_state=42, stratify=temp_df["label"])

split_map = {fname: "train" for fname in train_df["filename"]}
split_map.update({fname: "val" for fname in val_df["filename"]})
split_map.update({fname: "test" for fname in test_df["filename"]})

for _, row in df.iterrows():
    fname = str(row["filename"])
    label = str(row["label"]).lower()
    split = split_map[fname]

    video_path = os.path.join(videos_dir, fname)
    if not os.path.exists(video_path):
        print(f"Missing: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pick random frame index for n/z
    rand_frame_idx = random.randint(0, total_frames - 1) if label in {"n", "z"} else None

    frame_idx = 0
    saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize
        h, w = frame.shape[:2]
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Save frames
        if label == "c":
            out_path = os.path.join(output_dir, split, label, f"{Path(fname).stem}_frame{frame_idx}.jpg")
            cv2.imwrite(out_path, resized)

        elif label in {"n", "z"} and frame_idx == rand_frame_idx and not saved:
            out_path = os.path.join(output_dir, split, label, f"{Path(fname).stem}_rand.jpg")
            cv2.imwrite(out_path, resized)
            saved = True

        frame_idx += 1

    cap.release()
    #print(f"Processed: {fname} -> {split}/{label}")
