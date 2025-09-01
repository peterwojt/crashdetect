import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader



class RandomPadTo224Tensor:
    def __init__(self, fill=0):
        self.fill = fill  # padding color (0 = black)

    def __call__(self, img):
        # img is a tensor: (C, H, W)
        c, h, w = img.shape
        target_size = 224

        # If already at least 224×224, just resize
        if h >= target_size and w >= target_size:
            return torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False
            ).squeeze(0)

        # Compute padding amounts
        pad_h = max(target_size - h, 0)
        pad_w = max(target_size - w, 0)

        # Random split
        top = random.randint(0, pad_h)
        bottom = pad_h - top
        left = random.randint(0, pad_w)
        right = pad_w - left

        # Apply padding
        img = torch.nn.functional.pad(img, (left, right, top, bottom), value=self.fill)

        # Final resize in case one side was bigger than 224
        if img.shape[1] != target_size or img.shape[2] != target_size:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False
            ).squeeze(0)

        return img





# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data transforms (augmentations + normalization)
data_transforms = {
    "train": transforms.Compose([
        transforms.ToTensor(),          # convert PIL → tensor (0..1)
        RandomPadTo224Tensor(fill=0),   # now pad on tensor
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        RandomPadTo224Tensor(fill=0),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([   # test = same as val
        transforms.ToTensor(),
        RandomPadTo224Tensor(fill=0),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# 3. Load dataset from folders
# Folder structure should be:
# data/
#   train/
#       crash/
#       no_crash/
#   val/
#       crash/
#       no_crash/

data_dir = "data"  # change to your dataset path
image_datasets = {
    x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])
    for x in ["train", "val", "test"]
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=(x=="train"), num_workers=2)
    for x in ["train", "val", "test"]
}

# 4. Load pretrained ResNet18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: crash / no_crash
model = model.to(device)

# 5. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 20)

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

print("Training complete!")

# 7. Evaluate on test set
print("\nEvaluating on TEST set...")
model.eval()
running_corrects = 0

with torch.no_grad():
    for inputs, labels in dataloaders["test"]:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

test_acc = running_corrects.double() / len(image_datasets["test"])
print(f"TEST Accuracy: {test_acc:.4f}")

# 8. Save model
torch.save(model.state_dict(), "resnet18_crash.pth")

# 9. Example inference
def predict_image(image_path, model, transform):
    from PIL import Image
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    return "crash" if pred.item() == 0 else "no_crash"

#print(predict_image("test.jpg", model, data_transforms["test"]))