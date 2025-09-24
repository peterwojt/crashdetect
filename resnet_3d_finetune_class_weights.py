import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_video
import torch.nn.functional as F
import random



# Paths
DATA_DIR = "car_crash_video_dataset2"
BATCH_SIZE = 2
NUM_EPOCHS = 80
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAMES_PER_CLIP = 16  # Number of frames per video clip



# Class distribution
num_non_crash = 13553
num_crash = 684
total = num_non_crash + num_crash

# Inverse frequency weighting
weight_non_crash = total / (2 * num_non_crash)
weight_crash = total / (2 * num_crash)

# Class weights tensor
class_weights = torch.tensor([weight_non_crash, weight_crash], dtype=torch.float).to(DEVICE)





# Transforms for video frames
transform = transforms.Compose([
    transforms.Resize(112),           # Resize shortest side to 112, keep aspect ratio
    transforms.RandomCrop(112),       # Randomly crop to 112x112
    transforms.ToTensor(),
])

color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

def tensor_frame_transform(frame, size=112, spatial_aug=True):
    frame = frame.permute(2, 0, 1).float() / 255.0  # (C,H,W)
    frame = F.interpolate(frame.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False).squeeze(0)
    
    if spatial_aug:
        if random.random() > 0.5:
            frame = torch.flip(frame, dims=[2])
        # Apply color jitter (convert to PIL and back or implement in tensor)
        frame = color_jitter(frame)
    return frame


def temporal_augmentation(video, frames_per_clip=8):
    total_frames = video.shape[0]

    if total_frames <= frames_per_clip:
        return video  # no temporal aug if too short

    # Random start index to sample consecutive frames (temporal jitter)
    start_idx = random.randint(0, total_frames - frames_per_clip)
    sampled_frames = video[start_idx:start_idx + frames_per_clip]

    return sampled_frames

class SimpleVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=8, spatial_aug=False, temporal_aug=False, num_aug_crash=0, num_aug_no_crash=0):
        self.samples = []
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.spatial_aug = spatial_aug
        self.temporal_aug = temporal_aug

        for label in ['crash', 'non_crash']:
            class_dir = os.path.join(root_dir, label)

            label_idx = 1 if label == 'crash' else 0

            if label == 'crash':
                num_augmentations = num_aug_crash
            else:
                num_augmentations = num_aug_no_crash

            for fname in os.listdir(class_dir):
                if fname.endswith('.mp4'):

                    self.samples.append((os.path.join(class_dir, fname), 0 if label == 'non_crash' else 1))
                    for _ in range(num_augmentations):
                        self.samples.append((os.path.join(class_dir, fname), 0 if label == 'non_crash' else 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video, _, _ = read_video(video_path, pts_unit='sec')

        # Apply temporal augmentation if enabled
        if self.temporal_aug:
            video = temporal_augmentation(video, self.frames_per_clip)

        # Pad if less frames than needed
        if video.shape[0] < self.frames_per_clip:
            pad = self.frames_per_clip - video.shape[0]
            video = torch.cat([video, video[:1].repeat(pad, 1, 1, 1)], dim=0)
        else:
            video = video[:self.frames_per_clip]

        # Apply spatial augmentations per frame
        frames = []
        for frame in video:
            frame = tensor_frame_transform(frame, size=112, spatial_aug=self.spatial_aug)
            frames.append(frame)
        video = torch.stack(frames)

        # Rearrange to (C, T, H, W)
        video = video.permute(1, 0, 2, 3)
        return video, label

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss > self.best_loss - self.min_delta:  # no improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop




# Remove transforms.ToTensor() from your transform pipeline
transform = None

# Datasets and loaders
train_dataset = SimpleVideoDataset(
    os.path.join(DATA_DIR, "train"),
    transform=None,
    frames_per_clip=FRAMES_PER_CLIP,
    spatial_aug=True,
    temporal_aug=True,
    num_aug_crash=3,
    num_aug_no_crash=0
)

val_dataset = SimpleVideoDataset(
    os.path.join(DATA_DIR, "val"),
    transform=None,
    frames_per_clip=FRAMES_PER_CLIP,
    spatial_aug=False,
    temporal_aug=False,
)

test_dataset = SimpleVideoDataset(
    os.path.join(DATA_DIR, "test"),
    transform=None,
    frames_per_clip=FRAMES_PER_CLIP,
    spatial_aug=False,
    temporal_aug=False,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pretrained 3D ResNet
model = torchvision.models.video.r2plus1d_18(pretrained=True)

#model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Replace fc with dropout + linear
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

for name, param in model.named_parameters():
    #if "layer4" not in name and "fc" not in name:
    if "fc" not in name:
        param.requires_grad = False
    
    else:
        param.requires_grad = True

for name, param in model.layer4[1].conv2.named_parameters():
    param.requires_grad = True

for name, param in model.named_parameters():
    print(name, param.requires_grad)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

early_stopping = EarlyStopping(patience=8, min_delta=0.0)  # tune these values

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 5e-5},
    {"params": model.fc.parameters(), "lr": 5e-5}
], weight_decay=1e-4)

best_val_loss = 10.0

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_batches = len(train_loader)
    correct = 0
    total = 0
    running_loss = 0.0

    # Class-wise counters (TRAIN)
    train_class_correct = [0, 0]
    train_class_total = [0, 0]

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    for batch_idx, (videos, labels) in enumerate(train_loader, start=1):
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        for i in range(len(labels)):
            label = labels[i].item()
            pred = predicted[i].item()
            train_class_total[label] += 1
            if label == pred:
                train_class_correct[label] += 1

        batches_left = total_batches - batch_idx
        current_accuracy = (correct / total) * 100
        print(f"Batch {batch_idx}/{total_batches} — {batches_left} left — Accuracy: {current_accuracy:.2f}%", end='\r')

    # End of epoch
    epoch_acc = correct / total * 100
    avg_loss = running_loss / total_batches
    print(f"\nEpoch {epoch + 1} completed — Avg Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    print("Train class-wise correct predictions:")
    for cls in [0, 1]:
        acc = 100 * train_class_correct[cls] / train_class_total[cls] if train_class_total[cls] > 0 else 0.0
        print(f"  Class {cls} — {train_class_correct[cls]}/{train_class_total[cls]} correct ({acc:.2f}%)")

    # === Validation ===
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    val_class_correct = [0, 0]
    val_class_total = [0, 0]

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                val_class_total[label] += 1
                if label == pred:
                    val_class_correct[label] += 1

    val_acc = val_correct / val_total * 100
    val_avg_loss = val_loss / len(val_loader)

    print(f"Validation — Avg Loss: {val_avg_loss:.4f}, Accuracy: {val_acc:.2f}%")

    print("Validation class-wise correct predictions:")
    for cls in [0, 1]:
        acc = 100 * val_class_correct[cls] / val_class_total[cls] if val_class_total[cls] > 0 else 0.0
        print(f"  Class {cls} — {val_class_correct[cls]}/{val_class_total[cls]} correct ({acc:.2f}%)")
    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        torch.save(model.state_dict(), "r2plus1d18_best.pth")
        print(f"Best model updated at epoch {epoch+1}, val_acc={val_acc:.2f}%")

    if early_stopping(val_loss):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

print("Finetuning complete.")


model.eval()
test_correct = 0
test_total = 0
test_loss = 0.0
test_class_correct = [0, 0]
test_class_total = [0, 0]

with torch.no_grad():
    for videos, labels in test_loader:
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(videos)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

        for i in range(len(labels)):
            label = labels[i].item()
            pred = predicted[i].item()
            test_class_total[label] += 1
            if label == pred:
                test_class_correct[label] += 1

test_acc = test_correct / test_total * 100
test_avg_loss = test_loss / len(test_loader)
print(f"Test — Avg Loss: {test_avg_loss:.4f}, Accuracy: {test_acc:.2f}%")

print("Test class-wise correct predictions:")
for cls in [0, 1]:
    acc = 100 * test_class_correct[cls] / test_class_total[cls] if test_class_total[cls] > 0 else 0.0
    print(f"  Class {cls} — {test_class_correct[cls]}/{test_class_total[cls]} correct ({acc:.2f}%)")
