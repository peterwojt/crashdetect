import os
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import s3d, S3D_Weights
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Dataset
# -----------------------------
class FlowDataset(Dataset):
    def __init__(self, root_dir, split="train", max_frames=16):
        """
        root_dir: e.g., "data/"
        split: "train", "val", or "test"
        """
        self.samples = []
        self.max_frames = max_frames

        for label, cls in enumerate(["non_crash", "crash"]):  # label 0 = no_crash, 1 = crash
            split_folder = os.path.join(root_dir, cls, split)
            if not os.path.exists(split_folder):
                continue
            for f in os.listdir(split_folder):
                if f.endswith(".npy"):
                    self.samples.append((os.path.join(split_folder, f), label))

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        flow = np.load(path)  # (T,H,W,2)
        T, H, W, C = flow.shape

        # Target size
        target_h, target_w = 224, 224
        flow_resized = np.zeros((T, target_h, target_w, C), dtype=flow.dtype)

        # Compute scaling factor to preserve aspect ratio
        scale = min(target_h / H, target_w / W)
        new_h, new_w = int(H * scale), int(W * scale)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2

        # Resize each channel and pad
        for t in range(T):
            for c in range(C):
                resized = cv2.resize(flow[t,:,:,c], (new_w, new_h))
                flow_resized[t, top:top+new_h, left:left+new_w, c] = resized

        flow = flow_resized

        # Pad or trim in time dimension
        if T < self.max_frames:
            pad_shape = (self.max_frames - T, target_h, target_w, C)
            pad = np.zeros(pad_shape, dtype=flow.dtype)
            flow = np.concatenate([flow, pad], axis=0)
        elif T > self.max_frames:
            flow = flow[:self.max_frames]
        flow = flow.astype(np.float32)
        flow = flow / (np.max(np.abs(flow)) + 1e-8)

        # To tensor (C,T,H,W)
        flow = torch.from_numpy(flow).float().permute(3,0,1,2)
        return flow, torch.tensor(label).long()




# -----------------------------
# Data Loaders
# -----------------------------
train_dataset = FlowDataset("optical_flow_dataset2", split="train")
val_dataset   = FlowDataset("optical_flow_dataset2", split="val")
test_dataset  = FlowDataset("optical_flow_dataset2", split="test")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1)
test_loader  = DataLoader(test_dataset, batch_size=1)


# -----------------------------
# Model
# -----------------------------
weights = S3D_Weights.KINETICS400_V1
model = s3d(weights=weights)

for param in model.parameters():
    param.requires_grad = False
for name, param in model.features[15].named_parameters():
    if "branch" in name and "Conv3d" in name:
        param.requires_grad = True
# Modify first conv for 2-channel optical flow
# Correct for S3D
orig_conv = model.features[0][0][0]  # the first Conv3d
model.features[0][0][0] = nn.Conv3d(
    in_channels=2,                  # 2 channels for optical flow
    out_channels=orig_conv.out_channels,
    kernel_size=orig_conv.kernel_size,
    stride=orig_conv.stride,
    padding=orig_conv.padding,
    bias=False
)


# Replace fc for binary classification
#model.fc = nn.Linear(model.fc.in_features, 2)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Conv3d(1024, 2, kernel_size=(1,1,1))  # crash / no crash
)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# -----------------------------
# Optimizer & Loss
# -----------------------------
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# Training Loop
# -----------------------------
num_epochs = 10
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_train_preds, all_train_labels = [], []
    for i, (flows, labels) in enumerate(train_loader):
        flows, labels = flows.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(flows)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()       # batch predictions
        labels_np = labels.cpu().numpy()                  # batch labels
        all_train_preds.extend(preds)
        all_train_labels.extend(labels_np)

        # Accumulate for epoch-level accuracy so far
        all_train_preds.extend(preds)
        all_train_labels.extend(labels_np)
        epoch_acc_so_far = accuracy_score(all_train_labels, all_train_preds)

        # Print batch info + current epoch accuracy
        print(f"Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | "
              f"Loss: {loss.item():.4f} | Epoch Acc So Far: {epoch_acc_so_far:.4f}")

    avg_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for flows, labels in val_loader:
            flows, labels = flows.to(device), labels.to(device)
            outputs = model(flows)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    val_acc = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_s3d_flow.pth")

# -----------------------------
# Test Evaluation
# -----------------------------
model.load_state_dict(torch.load("best_s3d_flow.pth"))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for flows, labels in test_loader:
        flows, labels = flows.to(device), labels.to(device)
        outputs = model(flows)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

print("Test Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
