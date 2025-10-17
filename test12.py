import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -------------------
# 1️⃣ Load features + labels
# -------------------
data = np.load("video_features_labels.npz")
X = data['features']        # shape (num_videos, feature_dim)
y = data['labels']          # 0=c, 1=n

# -------------------
# 2️⃣ Split train/test
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------
# 3️⃣ Oversample minority class in training
# -------------------
class_counts = np.bincount(y_train)
max_count = max(class_counts)

indices_c = np.where(y_train == 0)[0]
indices_n = np.where(y_train == 1)[0]

np.random.seed(42)
if len(indices_c) < max_count:
    indices_c_upsampled = np.random.choice(indices_c, max_count, replace=True)
else:
    indices_c_upsampled = indices_c

if len(indices_n) < max_count:
    indices_n_upsampled = np.random.choice(indices_n, max_count, replace=True)
else:
    indices_n_upsampled = indices_n

balanced_indices = np.concatenate([indices_c_upsampled, indices_n_upsampled])
np.random.shuffle(balanced_indices)

X_train_bal = X_train[balanced_indices]
y_train_bal = y_train[balanced_indices]

# Convert to tensors
X_train_bal = torch.tensor(X_train_bal, dtype=torch.float32)
y_train_bal = torch.tensor(y_train_bal, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# -------------------
# 4️⃣ Dataset & DataLoader
# -------------------
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(FeatureDataset(X_train_bal, y_train_bal), batch_size=16, shuffle=True)
test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=16, shuffle=False)

# -------------------
# 5️⃣ Small hidden layer binary classifier
# -------------------
class SmallHiddenBinary(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)  # single output
        )
        
    def forward(self, x):
        return self.net(x)  # BCEWithLogitsLoss applies sigmoid

input_dim = X_train.shape[1]
model = SmallHiddenBinary(input_dim=input_dim, hidden_dim=32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------
# 6️⃣ Loss & optimizer
# -------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------
# 7️⃣ Training loop
# -------------------
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# -------------------
# 8️⃣ Evaluation
# -------------------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['c','n']))

# -------------------
# 9️⃣ Save model
# -------------------
torch.save(model.state_dict(), "small_hidden_binary.pth")
print("✅ Model saved to small_hidden_binary.pth")
