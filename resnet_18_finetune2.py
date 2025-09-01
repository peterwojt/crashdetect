import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models, transforms
import cv2
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve
)
# ------------------------
# Random padding transform
# ------------------------
class RandomPadTo224Tensor:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        c, h, w = img.shape
        target_size = 224

        if h >= target_size and w >= target_size:
            return F.interpolate(img.unsqueeze(0), size=(target_size, target_size),
                                 mode="bilinear", align_corners=False).squeeze(0)

        pad_h = max(target_size - h, 0)
        pad_w = max(target_size - w, 0)
        top = random.randint(0, pad_h)
        bottom = pad_h - top
        left = random.randint(0, pad_w)
        right = pad_w - left

        img = F.pad(img, (left, right, top, bottom), value=self.fill)

        if img.shape[1] != target_size or img.shape[2] != target_size:
            img = F.interpolate(img.unsqueeze(0), size=(target_size, target_size),
                                mode="bilinear", align_corners=False).squeeze(0)
        return img

# ------------------------
# Load image as tensor
# ------------------------
def load_image_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return img

# ------------------------
# Dataset that returns file paths
# ------------------------
class PathDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # idx can be int (from DataLoader) or str (from custom sampler)
        if isinstance(idx, int):
            return self.file_list[idx]
        return idx

# ------------------------
# Custom batch sampler for 25% c, 50% n, 25% z
# ------------------------
class DistributionBatchSampler(Sampler):
    def __init__(self, c_files, n_files, z_files, batch_size=32):
        self.c_files = c_files.copy()
        self.n_files = n_files.copy()
        self.z_files = z_files.copy()
        self.batch_size = batch_size

        self.batch_c = batch_size // 4
        self.batch_z = batch_size // 4
        self.batch_n = batch_size - self.batch_c - self.batch_z

    def __iter__(self):
        c_pool = self.c_files.copy()
        n_pool = self.n_files.copy()
        z_pool = self.z_files.copy()
        while len(c_pool) > 0:
            c_batch = [c_pool.pop(0) for _ in range(min(self.batch_c, len(c_pool)))]
            n_batch = random.sample(n_pool, min(self.batch_n, len(n_pool)))
            z_batch = random.sample(z_pool, min(self.batch_z, len(z_pool)))
            batch = c_batch + n_batch + z_batch
            random.shuffle(batch)
            yield batch


    def __len__(self):
        return (len(self.c_files) + self.batch_c - 1) // self.batch_c

# ------------------------
# Collate function to load tensors
# ------------------------
def collate_batch(batch_paths, transform):
    imgs, labels = [], []
    for path in batch_paths:
        img = load_image_tensor(path)
        if transform:
            img = transform(img)
        label = 0 if "/c/" in path else 1  # c=0, n/z=1
        imgs.append(img)
        labels.append(label)
    return torch.stack(imgs), torch.tensor(labels)

# ------------------------
# Get file lists
# ------------------------
def get_file_lists(split):
    base = f"data/{split}"
    c_files = [os.path.join(base, "c", f) for f in os.listdir(os.path.join(base, "c"))]
    n_files = [os.path.join(base, "n", f) for f in os.listdir(os.path.join(base, "n"))]
    z_files = [os.path.join(base, "z", f) for f in os.listdir(os.path.join(base, "z"))]
    return c_files, n_files, z_files

# ------------------------
# Device and transforms
# ------------------------
device = torch.device("cpu")  # CPU-only
transform = transforms.Compose([
    RandomPadTo224Tensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------------
# Create DataLoader
# ------------------------
def create_loader(split, batch_size=32):
    c_files, n_files, z_files = get_file_lists(split)
    all_files = c_files + n_files + z_files
    dataset = PathDataset(all_files)
    sampler = DistributionBatchSampler(c_files, n_files, z_files, batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        collate_fn=lambda batch: collate_batch(batch, transform),
                        num_workers=0)
    return loader

train_loader = create_loader("train")
val_loader = create_loader("val")
test_loader = create_loader("test")

# ------------------------
# Load pretrained ResNet18
# ------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)



# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only layer4 + fc
for name, param in model.named_parameters():
    if name.startswith("layer4") or name.startswith("fc"):
        param.requires_grad = True



num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# Example: more weight to crash class
weights = torch.tensor([3.0, 1.0])  # crash=3x, no crash=1x
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_c, train_n, train_z = get_file_lists("train")
val_c, val_n, val_z = get_file_lists("val")
train_size = len(train_c + train_n + train_z)
val_size = len(val_c + val_n + val_z)

# ------------------------
# Training loop
# ------------------------

num_epochs = 150
patience = 10   # stop if val loss doesn’t improve for 5 epochs

best_val_loss = float("inf")
epochs_no_improve = 0
best_epoch = -1
csv_file = "model_metrics_log.csv"

# Count how many entries (rows) are in the CSV
num_entries = 0
if os.path.isfile(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        num_entries = sum(1 for _ in reader)

# New model checkpoint name based on entries
checkpoint_path = f"models/model_{num_entries+1}.pth"

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-"*20)

    for phase in ["train", "val"]:
        loader = train_loader if phase=="train" else val_loader
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0

        all_preds, all_labels, all_probs = [], [], []

        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase=="train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs,1)

                if phase=="train":
                    loss.backward()
                    optimizer.step()

            # Collect metrics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

            # TP/FP/TN/FN (assuming class 0 = positive, class 1 = negative; adjust if swapped)
            tp = ((preds == 0) & (labels == 0)).sum().item()
            fp = ((preds == 0) & (labels == 1)).sum().item()
            tn = ((preds == 1) & (labels == 1)).sum().item()
            fn = ((preds == 1) & (labels == 0)).sum().item()
            tp_total += tp
            fp_total += fp
            tn_total += tn
            fn_total += fn

            # Save for ROC/PR
            probs = F.softmax(outputs, dim=1)[:,0]  # prob of class 0 (positive)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            batch_acc = torch.sum(preds == labels.data).double() / labels.size(0)
            print(f"{phase} Batch {batch_idx+1}/{len(loader)}: "
                  f"Loss {loss.item():.4f} Acc {batch_acc:.4f} "
                  f"| TP={tp} FP={fp} FN={fn} TN={tn}    ", end="\r")

        dataset_size = train_size if phase=="train" else val_size
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / float(dataset_size)

        # ---- Compute extra metrics ----
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall    = recall_score(all_labels, all_preds, zero_division=0)
        f1        = f1_score(all_labels, all_preds, zero_division=0)

        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            roc_auc = float("nan")  # if only one class predicted

        pr_auc = average_precision_score(all_labels, all_probs)

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
              f"| TP={tp_total} FP={fp_total} FN={fn_total} TN={tn_total}")
        print(f"{phase} Precision: {precision:.4f} Recall: {recall:.4f} "
              f"F1: {f1:.4f} ROC-AUC: {roc_auc:.4f} PR-AUC: {pr_auc:.4f}")

        # -------- Early stopping check --------
        if phase == "val":
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}, best was {best_epoch+1}")
                    model.load_state_dict(torch.load(checkpoint_path))
                    break
    else:
        continue
    break
# ------------------------
# Test evaluation
# ------------------------
model.eval()

all_preds = []
all_labels = []
all_probs = []

c_files, n_files, z_files = get_file_lists("test")
test_dataset_size = len(c_files + n_files + z_files)

for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)                # raw logits
        probs = torch.softmax(outputs, dim=1)  # convert to probabilities
        _, preds = torch.max(outputs,1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())
    all_probs.extend(probs[:,1].cpu().numpy())  # prob for class "1" (positive)

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

# Metrics
accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall    = recall_score(all_labels, all_preds, zero_division=0)
f1        = f1_score(all_labels, all_preds, zero_division=0)

roc_auc   = roc_auc_score(all_labels, all_probs)
pr_auc    = average_precision_score(all_labels, all_probs)

print("TEST RESULTS")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

results = {
    "model_name": checkpoint_path,
    "accuracy":  accuracy,
    "precision": precision,
    "recall":    recall,
    "f1":        f1,
    "roc_auc":   roc_auc,
    "pr_auc":    pr_auc,
    "tp":        tp,
    "fp":        fp,
    "fn":        fn,
    "tn":        tn,
}

# -----------------------
# Append results to CSV
# -----------------------
file_exists = os.path.isfile(csv_file)
with open(csv_file, mode="a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results.keys())
    if not file_exists:
        writer.writeheader()  # write header first time
    writer.writerow(results)

print(f"Logged results for {results['model_name']}")
print(f"Next checkpoint will be saved as: {checkpoint_path}")
