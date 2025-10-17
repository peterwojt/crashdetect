import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------
# Paths
# -------------------
video_dir = Path("cropped_crash_videos2")
csv_file = Path("labels2.csv")
output_file = Path("video_features_labels.npz")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Load pretrained Video Swin
# -------------------
model = torchvision.models.video.swin3d_t(pretrained=True)
model.eval()
model.to(device)
feature_extractor = nn.Sequential(*list(model.children())[:-1])

# -------------------
# Read labels CSV
# -------------------
df_labels = pd.read_csv(csv_file)
df_labels = df_labels[df_labels['label'].isin(['c', 'n'])]  # filter if needed

# -------------------
# Helper functions
# -------------------
def read_video_frames(video_path, num_frames=16, resize=(224,224)):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return np.array([])
    
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames)

def preprocess_frames(frames):
    frames = frames.astype(np.float32) / 255.0
    frames = frames.transpose(3,0,1,2)  # (C,T,H,W)
    frames = torch.tensor(frames).unsqueeze(0)
    return frames

# -------------------
# Extract features for all videos
# -------------------
all_features = []
all_labels = []

for _, row in df_labels.iterrows():
    video_path = video_dir / row['filename']
    label = row['label']
    
    if not video_path.exists():
        print(f"Warning: {video_path} not found.")
        continue
    
    frames = read_video_frames(video_path)
    if len(frames) == 0:
        print(f"Warning: no frames in {video_path}")
        continue
    
    x = preprocess_frames(frames).to(device)
    with torch.no_grad():
        features = feature_extractor(x)
        features = features.flatten().cpu().numpy()
    
    all_features.append(features)
    all_labels.append(0 if label=='c' else 1)

    print(f"Processed {video_path}")

# -------------------
# Save all to one file
# -------------------
all_features = np.stack(all_features)
all_labels = np.array(all_labels)
np.savez(output_file, features=all_features, labels=all_labels)
print(f"Saved all features and labels to {output_file}")
