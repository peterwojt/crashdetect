import os
import cv2
import pandas as pd
from pathlib import Path
import shutil

# -------- CONFIG --------
videos_dir = "videos"         # folder containing numbered videos
csv_file = "labels.csv"       # CSV with "number,label"
output_dir = "data"  # root folder for sorted videos
valid_labels = {"c", "z", "n"}  # which labels to keep
max_size = 224                # maximum height/width
# -------------------------

# Create output folders
for label in valid_labels:
    Path(output_dir, label).mkdir(parents=True, exist_ok=True)

# Read CSV
df = pd.read_csv(csv_file)

for _, row in df.iterrows():
    num, label = str(row[0]), str(row[1]).lower()
    if label not in valid_labels:
        continue
    
    # input video path
    video_path = os.path.join(videos_dir, f"{num}.mp4")
    if not os.path.exists(video_path):
        print(f"Missing: {video_path}")
        continue

    # output video path
    out_path = os.path.join(output_dir, label, f"{num}.mp4")

    # Open video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute scaling
    scale = min(max_size / width, max_size / height)
    new_w, new_h = int(width * scale), int(height * scale)

    out = cv2.VideoWriter(out_path, fourcc, fps, (new_w, new_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out.write(resized)

    cap.release()
    out.release()

    #print(f"Saved: {out_path}")
