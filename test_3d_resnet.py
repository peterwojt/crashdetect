import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np

# Load pre-trained 3D ResNet model (Kinetics-400)
model = torchvision.models.video.r3d_18(pretrained=True)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])

def load_video_cv2(path, num_frames=16):
    cap = cv2.VideoCapture(path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    idx = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i == frame_indices[idx]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR -> RGB
            frame = transforms.ToPILImage()(frame)
            frame = transform(frame)
            frames.append(frame)
            idx += 1
            if idx >= len(frame_indices):
                break

    cap.release()
    video = torch.stack(frames)  # [T, C, H, W]
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
    return video

# Path to your video
video_path = "videos/522.mp4"
video_tensor = load_video_cv2(video_path)

# Predict
with torch.no_grad():
    outputs = model(video_tensor)
    pred_class = outputs.argmax(-1).item()

print("Predicted class number:", pred_class)



# outputs from model
logits = outputs  # shape: [1, 400]

# Convert to probabilities
probs = F.softmax(logits, dim=1)  # shape: [1, 400]
pred_index = probs.argmax(-1).item()
pred_confidence = probs[0, pred_index].item()

print(f"Predicted class number: {pred_index}")
print(f"Predicted confidence: {pred_confidence:.2f}")


# Predict
#with torch.no_grad():
#    outputs = model(video_tensor)
#    pred_class = outputs.argmax(-1).item()

# Kinetics-400 label mapping
#kinetics_classes = torchvision.datasets.Kinetics400.classes
#print("Predicted action:", kinetics_classes[pred_class])
