import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

# ====== CONFIG ======
DATA_DIR = "car_crash_video_dataset2"
BATCH_SIZE = 4
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAMES_PER_CLIP = 16

# ====== Frame Transform ======
import torchvision.transforms as transforms
color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

def tensor_frame_transform(frame, size=112, spatial_aug=False):
    frame = frame.permute(2, 0, 1).float() / 255.0  # (C,H,W)
    frame = F.interpolate(frame.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False).squeeze(0)
    if spatial_aug:
        if random.random() > 0.5:
            frame = torch.flip(frame, dims=[2])
        frame = color_jitter(frame)
    return frame

def temporal_augmentation(video, frames_per_clip=8):
    total_frames = video.shape[0]
    if total_frames <= frames_per_clip:
        return video
    start_idx = random.randint(0, total_frames - frames_per_clip)
    return video[start_idx:start_idx + frames_per_clip]

# ====== Dataset ======
class SimpleVideoDataset(Dataset):
    def __init__(self, root_dir, frames_per_clip=8, spatial_aug=False, temporal_aug=False):
        self.samples = []
        self.frames_per_clip = frames_per_clip
        self.spatial_aug = spatial_aug
        self.temporal_aug = temporal_aug

        for label in ['crash', 'non_crash']:
            class_dir = os.path.join(root_dir, label)
            for fname in os.listdir(class_dir):
                if fname.endswith('.mp4'):
                    self.samples.append((os.path.join(class_dir, fname), 0 if label == 'non_crash' else 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video, _, _ = read_video(video_path, pts_unit='sec')

        if self.temporal_aug:
            video = temporal_augmentation(video, self.frames_per_clip)

        if video.shape[0] < self.frames_per_clip:
            pad = self.frames_per_clip - video.shape[0]
            video = torch.cat([video, video[:1].repeat(pad, 1, 1, 1)], dim=0)
        else:
            video = video[:self.frames_per_clip]

        frames = [tensor_frame_transform(frame, size=112, spatial_aug=self.spatial_aug) for frame in video]
        video = torch.stack(frames)

        # Rearrange to (C, T, H, W)
        video = video.permute(1, 0, 2, 3)
        return video, label

# ====== Test Loader ======
test_dataset = SimpleVideoDataset(
    os.path.join(DATA_DIR, "test"),
    frames_per_clip=FRAMES_PER_CLIP,
    spatial_aug=False,
    temporal_aug=False,
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====== Model ======
model = torchvision.models.video.r2plus1d_18(pretrained=False)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

# Load best weights
model.load_state_dict(torch.load("r2plus1d18_best1.pth", map_location=DEVICE))
model.eval()

# ====== Evaluation ======
from collections import Counter
import sys

# ====== Live Evaluation (One-Line Print) ======
all_preds = []
all_labels = []
all_video_ids = []

TP = TN = FP = FN = 0
class_correct = Counter()
class_total = Counter()
total_seen = 0
total_correct = 0

with torch.no_grad():
    for batch_idx, (videos, labels) in enumerate(test_loader):
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(videos)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)

        batch_preds = predicted.cpu().numpy()
        batch_labels = labels.cpu().numpy()
        all_preds.extend(batch_preds)
        all_labels.extend(batch_labels)

        # Track video paths
        start_idx = batch_idx * BATCH_SIZE
        batch_video_paths = [test_dataset.samples[i + start_idx][0] for i in range(len(labels))]
        all_video_ids.extend(batch_video_paths)

        # Update stats
        for i in range(len(batch_labels)):
            true = batch_labels[i]
            pred = batch_preds[i]

            total_seen += 1
            if true == pred:
                total_correct += 1
                if true == 1:
                    TP += 1
                else:
                    TN += 1
                class_correct[true] += 1
            else:
                if pred == 1 and true == 0:
                    FP += 1
                elif pred == 0 and true == 1:
                    FN += 1

            class_total[true] += 1

        # Accuracy
        accuracy = total_correct / total_seen
        class_0_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
        class_1_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0

        # One-line print
        sys.stdout.write(
            f"\rBatch {batch_idx + 1}/{len(test_loader)} | "
            f"Acc: {accuracy:.4f} | "
            f"non_crash Acc: {class_0_acc:.4f} | crash Acc: {class_1_acc:.4f} | "
            f"TP: {TP} TN: {TN} FP: {FP} FN: {FN}"
        )
        sys.stdout.flush()

# ====== Metrics ======
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)

softmax_probs = []

with torch.no_grad():
    for batch_idx, (videos, labels) in enumerate(test_loader):
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(videos)
        
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        softmax_probs.extend(probs)

print("\n=== Test Results ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

print("\n=== Per-class metrics ===")
print(classification_report(all_labels, all_preds, target_names=["non_crash", "crash"], digits=4))

# ====== Misclassified Videos ======
print("\n=== Misclassified video IDs (with probabilities) ===")

for i, (vid, true, pred, probs) in enumerate(zip(all_video_ids, all_labels, all_preds, softmax_probs)):
    if true != pred:
        probs = np.array(probs).flatten()  # ensure it's 1D
        prob_str = f"[non_crash: {probs[0]:.4f}, crash: {probs[1]:.4f}]"
        print(f"{os.path.basename(vid)} — True: {true}, Pred: {pred}, Probabilities: {prob_str}")
