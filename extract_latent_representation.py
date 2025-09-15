import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from autoencoder import Conv3DAutoencoder, FLOW_MIN, FLOW_MAX
import torch.nn.functional as F
import glob
import random
import torchvision.transforms.functional as TF

# -----------------------------
# Dataset with augmentation
# -----------------------------
class NpyFolderDataset(Dataset):
    def __init__(self, folder, target_size=(256,256), target_frames=16, augment=False):
        self.files = sorted(glob.glob(os.path.join(folder, "**", "*.npy"), recursive=True))
        self.target_size = target_size
        self.target_frames = target_frames
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        flow = np.load(self.files[idx])  # (T, H, W, 2)
        flow = (flow - FLOW_MIN) / (FLOW_MAX - FLOW_MIN + 1e-6)
        flow = np.transpose(flow, (3, 0, 1, 2))  # (C=2, T, H, W)
        flow = torch.from_numpy(flow).float()
        C, T, H, W = flow.shape

        # --- Temporal pad/truncate ---
        if T < self.target_frames:
            pad_T = self.target_frames - T
            pad_tensor = torch.zeros((C, pad_T, H, W))
            flow = torch.cat([flow, pad_tensor], dim=1)
        elif T > self.target_frames:
            flow = flow[:, :self.target_frames, :, :]

        # --- Data augmentation ---
        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                flow = torch.flip(flow, dims=[3])
                flow[0] = -flow[0]

            # Small rotation ±10°
            angle = random.uniform(-10, 10)
            flow = TF.rotate(flow, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)

            # Temporal jitter ±2 frames
            shift = random.randint(-2, 2)
            if shift > 0:
                flow = torch.cat([flow[:, shift:], flow[:, :shift]], dim=1)
            elif shift < 0:
                shift = abs(shift)
                flow = torch.cat([flow[:, -shift:], flow[:, :-shift]], dim=1)

        # --- Resize + padding ---
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
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        flow_padded = F.pad(flow_resized, (pad_left, pad_right, pad_top, pad_bottom))

        return flow_padded, self.files[idx]


# -----------------------------
# Latent extraction + save augmented
# -----------------------------
def extract_latents_with_augmentation(model_path="autoencoder.pth",
                                      folder="optical_flow_dataset/crash",
                                      batch_size=1,
                                      save_path="latents.npy",
                                      flatten=True,
                                      num_augment=3):  # how many augmentations per file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = Conv3DAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare dataset loader without augmentation first for originals
    dataset = NpyFolderDataset(folder, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_latents = []
    file_list = []

    with torch.no_grad():
        for data, filenames in loader:
            data = data.to(device)
            latents = model.encoder(data)
            latents = latents.max(dim=2).values
            latents = latents.max(dim=2).values
            latents = latents.max(dim=2).values
            if flatten:
                latents = latents.view(latents.size(0), -1)
            all_latents.append(latents.cpu().numpy())
            file_list.extend(filenames)

            # --- Augmentations ---
            for _ in range(num_augment):
                aug_dataset = NpyFolderDataset(folder, augment=True)
                # Pick the same batch size items for consistency
                aug_data, aug_filenames = aug_dataset.__getitem__(0)  # get one item at a time
                aug_data = aug_data.unsqueeze(0).to(device)
                aug_latents = model.encoder(aug_data)
                aug_latents = aug_latents.max(dim=2).values
                aug_latents = aug_latents.max(dim=2).values
                aug_latents = aug_latents.max(dim=2).values
                if flatten:
                    aug_latents = aug_latents.view(aug_latents.size(0), -1)
                all_latents.append(aug_latents.cpu().numpy())
                file_list.extend([f"{aug_filenames}_aug{i}" for i in range(num_augment)])

    all_latents = np.concatenate(all_latents, axis=0)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, all_latents)
    mapping_path = save_path.replace(".npy", "_files.txt")
    with open(mapping_path, "w") as f:
        f.write("\n".join(file_list))

    print(f"✅ Saved latents → {save_path}, shape = {all_latents.shape}")
    print(f"✅ File mapping → {mapping_path}")


if __name__ == "__main__":
    extract_latents_with_augmentation(
        model_path="autoencoder.pth",
        folder="flow_logistic_data/crash/train",
        batch_size=1,
        save_path="latent_data/crash_train.npy",
        flatten=True,
        num_augment=3
    )
