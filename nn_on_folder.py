import os
import csv
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
import torch.nn.functional as F
import random
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ====== CONFIG ======
DATA_DIR = "cropped_crash_videos2"
CSV_FILE = "labels2.csv"
BATCH_SIZE = 1
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAMES_PER_CLIP = 16

# ====== Load CSV labels ======
label_dict = {}
with open(CSV_FILE, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row["filename"].strip()
        label = row["label"].strip().lower()
        if label in ("c", "n"):
            label_dict[fname] = label

# ====== Frame Transform ======
color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

def tensor_frame_transform(frame, size=112, spatial_aug=False):
    frame = frame.permute(2, 0, 1).float() / 255.0
    frame = F.interpolate(frame.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False).squeeze(0)
    if spatial_aug:
        if random.random() > 0.5:
            frame = torch.flip(frame, dims=[2])
        frame = color_jitter(frame)
    return frame

# ====== Dataset ======
class SimpleVideoFolder(Dataset):
    def __init__(self, folder, frames_per_clip=16, label_dict=None):
        all_videos = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp4')]
        self.videos = [v for v in all_videos if os.path.basename(v) in label_dict]
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        path = self.videos[idx]
        video, _, _ = read_video(path, pts_unit='sec')

        if video.shape[0] < self.frames_per_clip:
            pad = self.frames_per_clip - video.shape[0]
            video = torch.cat([video, torch.zeros((pad, *video.shape[1:]), dtype=video.dtype)], dim=0)
        else:
            video = video[:self.frames_per_clip]

        frames = [tensor_frame_transform(f, size=112, spatial_aug=False) for f in video]
        video = torch.stack(frames).permute(1, 0, 2, 3)
        return video, path

# ====== Loader ======
dataset = SimpleVideoFolder(DATA_DIR, frames_per_clip=FRAMES_PER_CLIP, label_dict=label_dict)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====== Model ======
model = torchvision.models.video.r2plus1d_18(pretrained=False)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)
model.load_state_dict(torch.load("r2plus1d18_best1.pth", map_location=DEVICE))
model.eval()

# ====== Run Inference ======
all_true = []
all_pred = []

with torch.no_grad():
    for videos, paths in loader:
        videos = videos.to(DEVICE)
        outputs = model(videos)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        probs = probs.cpu().numpy()

        for path, pred, p in zip(paths, preds, probs):
            base = os.path.basename(path)
            true_label = label_dict[base]
            true_numeric = 1 if true_label == "c" else 0  # 1=crash, 0=non-crash
            all_true.append(true_numeric)
            all_pred.append(pred)
            pred_label = "CRASH" if pred == 1 else "NON-CRASH"
            print(f"{base} (true: {true_label}) -> {pred_label} [non_crash: {p[0]:.3f}, crash: {p[1]:.3f}]")

# ====== Metrics ======
accuracy = accuracy_score(all_true, all_pred)
precision = precision_score(all_true, all_pred)
recall = recall_score(all_true, all_pred)
f1 = f1_score(all_true, all_pred)
cm = confusion_matrix(all_true, all_pred)

print("\n===== METRICS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 score : {f1:.4f}")
print("Confusion Matrix:")
print(cm)
print("===================")
