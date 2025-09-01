import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models, transforms
import cv2

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
        while len(self.c_files) > 0:
            c_batch = [self.c_files.pop(0) for _ in range(min(self.batch_c, len(self.c_files)))]
            n_batch = random.sample(self.n_files, min(self.batch_n, len(self.n_files)))
            z_batch = random.sample(self.z_files, min(self.batch_z, len(self.z_files)))
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
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------
# Training loop
# ------------------------
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-"*20)

    for phase in ["train", "val"]:
        loader = train_loader if phase=="train" else val_loader
        model.train() if phase=="train" else model.eval()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase=="train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs,1)
                if phase=="train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        dataset_size = len(get_file_lists("train")[0] + get_file_lists("train")[1] + get_file_lists("train")[2]) \
            if phase=="train" else len(get_file_lists("val")[0] + get_file_lists("val")[1] + get_file_lists("val")[2])
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size
        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

# ------------------------
# Test evaluation
# ------------------------
model.eval()
running_corrects = 0
c_files, n_files, z_files = get_file_lists("test")
test_dataset_size = len(c_files + n_files + z_files)

for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs,1)
        running_corrects += torch.sum(preds == labels.data)

test_acc = running_corrects.double() / test_dataset_size
print(f"TEST Accuracy: {test_acc:.4f}")
