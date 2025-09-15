# autoencoder_optical_flow.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import random
import torch.nn.functional as F


FLOW_MIN = -361.9104
FLOW_MAX = 646.3912

class OpticalFlowDataset(Dataset):
    def __init__(self, root_dir, subset="train", target_size=(256,256), target_frames=16, transform=None):
        """
        root_dir: dataset root (optical_flow_dataset)
        subset: "train", "val", or "test"
        - train: only 'no_crash/train'
        - val/test: both 'crash/<subset>' and 'no_crash/<subset>'
        """
        self.files = []
        self.labels = []  # 0 = no_crash, 1 = crash

        if subset == "train":
            # only no_crash/train
            subset_path = os.path.join(root_dir, "no_crash", "train")
            for f in os.listdir(subset_path):
                if f.endswith(".npy"):
                    self.files.append(os.path.join(subset_path, f))
                    self.labels.append(0)

        else:
            # both crash and no_crash
            for category in ["no_crash", "crash"]:
                subset_path = os.path.join(root_dir, category, subset)
                if not os.path.isdir(subset_path):
                    continue
                for f in os.listdir(subset_path):
                    if f.endswith(".npy"):
                        self.files.append(os.path.join(subset_path, f))
                        self.labels.append(0 if category == "no_crash" else 1)

        self.target_size = target_size
        self.target_frames = target_frames
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        flow = np.load(self.files[idx])  # (T, H, W, 2)
        label = self.labels[idx]

        # normalize [0,1]
        flow = (flow - FLOW_MIN) / (FLOW_MAX - FLOW_MIN + 1e-6)

        # Rearrange (C=2, T, H, W)
        flow = np.transpose(flow, (3, 0, 1, 2))
        flow = torch.from_numpy(flow).float()

        # --- Temporal pad/truncate ---
        C, T, H, W = flow.shape
        if T < self.target_frames:
            pad_T = self.target_frames - T
            pad_tensor = torch.zeros((C, pad_T, H, W))
            flow = torch.cat([flow, pad_tensor], dim=1)
        elif T > self.target_frames:
            flow = flow[:, :self.target_frames, :, :]

        # --- Resize spatial dims with padding ---
        target_H, target_W = self.target_size
        scale = min(target_H / H, target_W / W)
        new_H, new_W = int(H * scale), int(W * scale)

        flow_resized = F.interpolate(
            flow.unsqueeze(0),
            size=(self.target_frames, new_H, new_W),
            mode="trilinear",
            align_corners=False
        ).squeeze(0)

        pad_H = target_H - new_H
        pad_W = target_W - new_W
        pad_top = random.randint(0, pad_H)
        pad_bottom = pad_H - pad_top
        pad_left = random.randint(0, pad_W)
        pad_right = pad_W - pad_left
        flow_padded = F.pad(flow_resized, (pad_left, pad_right, pad_top, pad_bottom))

        if self.transform:
            flow_padded = self.transform(flow_padded)

        return flow_padded, label



# ==============================
# Conv3D Autoencoder
# ==============================
class Conv3DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, stride=2, padding=1),  # -> (16, T/2, H/2, W/2)
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, T/4, H/4, W/4)
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, T/8, H/8, W/8)
            nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ==============================
# Training
# ==============================
# ==============================
# Training with Early Stopping
# ==============================
# ==============================
# Training with Early Stopping and Test Evaluation
# ==============================
# ==============================
# Training with Early Stopping and Separate Validation/Test for Crash/No Crash
# ==============================
def train_autoencoder(dataset_root="optical_flow_dataset",
                      batch_size=4,
                      lr=1e-3,
                      epochs=100,
                      save_path="autoencoder.pth",
                      patience=5):  # Early stopping patience

    train_dataset = OpticalFlowDataset(dataset_root, subset="train")
    val_dataset   = OpticalFlowDataset(dataset_root, subset="val")
    test_dataset  = OpticalFlowDataset(dataset_root, subset="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv3DAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0

        for i, (data, _) in enumerate(train_loader):
            data = data.to(device)
            outputs = model(data)

            if outputs.shape[2] != data.shape[2]:
                min_T = min(outputs.shape[2], data.shape[2])
                outputs = outputs[:, :, :min_T, :, :]
                data = data[:, :, :min_T, :, :]

            loss = criterion(outputs, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Batch [{i+1}/{len(train_loader)}] "
                f"Batch Loss: {loss.item():.4f} "
                f"Avg Loss: {avg_loss:.4f}",
                end="\r"
            )

        # --- Validation ---
        model.eval()
        val_loss_crash = 0.0
        val_loss_nocrash = 0.0
        count_crash = 0
        count_nocrash = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                outputs = model(data)
                if outputs.shape[2] != data.shape[2]:
                    min_T = min(outputs.shape[2], data.shape[2])
                    outputs = outputs[:, :, :min_T, :, :]
                    data = data[:, :, :min_T, :, :]

                for j in range(len(labels)):
                    l = labels[j].item()
                    sample_loss = criterion(outputs[j:j+1], data[j:j+1]).item()
                    if l == 0:
                        val_loss_nocrash += sample_loss
                        count_nocrash += 1
                    else:
                        val_loss_crash += sample_loss
                        count_crash += 1

        val_loss_crash /= max(count_crash, 1)
        val_loss_nocrash /= max(count_nocrash, 1)
        val_loss = (val_loss_crash + val_loss_nocrash) / 2

        print(f"\nEpoch [{epoch+1}/{epochs}] completed → "
              f"Avg Train Loss: {avg_loss:.4f} | "
              f"Val Loss NoCrash: {val_loss_nocrash:.4f} | "
              f"Val Loss Crash: {val_loss_crash:.4f}")

        # --- Early stopping check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)  # save best model
            print(f"Validation improved → model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} epochs → Early stopping")
                break

    # --- Load best model and test ---
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_loss_crash = 0.0
    test_loss_nocrash = 0.0
    count_crash = 0
    count_nocrash = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            if outputs.shape[2] != data.shape[2]:
                min_T = min(outputs.shape[2], data.shape[2])
                outputs = outputs[:, :, :min_T, :, :]
                data = data[:, :, :min_T, :, :]

            for j in range(len(labels)):
                l = labels[j].item()
                sample_loss = criterion(outputs[j:j+1], data[j:j+1]).item()
                if l == 0:
                    test_loss_nocrash += sample_loss
                    count_nocrash += 1
                else:
                    test_loss_crash += sample_loss
                    count_crash += 1

    test_loss_crash /= max(count_crash, 1)
    test_loss_nocrash /= max(count_nocrash, 1)

    print(f"Test completed → Avg Test Loss NoCrash: {test_loss_nocrash:.4f} | "
          f"Crash: {test_loss_crash:.4f}")
    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")



if __name__ == "__main__":
    train_autoencoder()
