import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2

# ---- Image preprocessing ----
def load_image_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return img

def preprocess(img_tensor):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # simple resize for inference
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transform(img_tensor)

# ---- Load trained model ----
def load_model(checkpoint_path, device="cpu"):
    model = models.resnet18(weights=None)      # same base architecture
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)          # same head as training
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ---- Run inference ----
def predict(image_path, checkpoint_path, device="cpu"):
    img = load_image_tensor(image_path)
    img = preprocess(img).unsqueeze(0).to(device)

    model = load_model(checkpoint_path, device)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = torch.argmax(probs).item()
    class_names = {0: "c (crash)", 1: "n/z (no-crash)"}
    print(f"Prediction: {class_names[pred_idx]}")
    print(f"Probabilities: crash={probs[0]:.4f}, no-crash={probs[1]:.4f}")

# Example usage
predict("example_images/example6.jpg", "models/model_1.pth")
