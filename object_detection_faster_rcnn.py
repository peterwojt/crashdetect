import cv2
import torch
import time

#from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
#from torchvision.models.detection import fcos_resnet50_fpn
#from torchvision.models.detection import retinanet_resnet50_fpn_v2

from torchvision import transforms
import numpy as np

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Faster R-CNN model
#model = fasterrcnn_resnet50_fpn(pretrained=True)
model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True).eval()
#model = retinanet_resnet50_fpn_v2(pretrained=True)
#model = fcos_resnet50_fpn(pretrained=True)

model.eval()
model.to(device)

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Video source (0 = webcam) or file path
#cap = cv2.VideoCapture('media_w1117040928_7.ts')
cap = cv2.VideoCapture('crashes/2023-2024/110_NE_4_-_Center_2024-04-18_20_18_19_042.mp4')
# Define preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Detection threshold
DETECTION_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time() 
    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image).to(device)



    # Forward pass
    with torch.no_grad():
        outputs = model([image_tensor])

    end_time = time.time()  # End timing

    # Calculate inference time in milliseconds
    inference_time_ms = (end_time - start_time) * 1000
    # Extract predictions
    pred = outputs[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()

    # Filter out detections below threshold
    for box, score, label in zip(boxes, scores, labels):
        if score < DETECTION_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.astype(int)
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        color = (0, 255, 0)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"Inference: {inference_time_ms:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # Show the frame
    cv2.imshow("Faster R-CNN Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
