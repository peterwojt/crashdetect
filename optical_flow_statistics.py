import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os

# paths
video_dir = Path("cropped_crash_videos2")
labels_csv = Path("labels2.csv")
output_csv = Path("video_flow_features.csv")

# load CSV
df = pd.read_csv(labels_csv)

# only keep c and n
df = df[df['label'].isin(['c', 'n'])]

# ensure output file has header if it doesn't exist
if not output_csv.exists():
    pd.DataFrame(columns=[
        "mean_magnitude",
        "peak_magnitude",
        "mean_variance",
        "peak_variance",
        "label"
    ]).to_csv(output_csv, index=False)

for _, row in df.iterrows():
    video_path = video_dir / row['filename']
    label = row['label']

    if not video_path.exists():
        print(f"Warning: {video_path} not found.")
        continue

    print(f"Processing {video_path} ...")

    cap = cv2.VideoCapture(str(video_path))
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Cannot read first frame from {video_path}")
        continue

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    mag_list = []
    var_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # compute optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)

        mag_list.append(np.mean(magnitude))
        var_list.append(np.var(angle))

        prev_gray = gray

    cap.release()

    if len(mag_list) == 0:
        print(f"No valid frames for {video_path}")
        continue

    mean_magnitude = np.median(mag_list)
    peak_magnitude = np.max(mag_list)
    mean_variance = np.median(var_list)
    peak_variance = np.max(var_list)

    # append result immediately
    result_df = pd.DataFrame([{
        "mean_magnitude": mean_magnitude,
        "peak_magnitude": peak_magnitude,
        "mean_variance": mean_variance,
        "peak_variance": peak_variance,
        "label": label
    }])

    # append without header
    result_df.to_csv(output_csv, mode='a', header=False, index=False)

    print(f"→ Saved results for {video_path}")

print(f"\n✅ All done! Results saved in {output_csv}")
