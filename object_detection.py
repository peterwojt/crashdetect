import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F
import cv2
import numpy as np

# Load the lightweight SSD model
model = ssd300_vgg16(pretrained=True).eval()

# COCO class labels for SSD (91 classes, index 0 is background)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Open video capture (0 for webcam, or filename)
#cap = cv2.VideoCapture('car_crash_video_dataset/train/crash/2110.mp4')  # Change to 'your_video.mp4' to read from a file
cap = cv2.VideoCapture('crashes/110_NE_4_-_Center_2024-04-18_20_18_19_042.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 300x300 for SSD input
    frame_resized = cv2.resize(frame, (300, 300))

    # Convert frame from BGR(OpenCV) to RGB and to tensor
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0)

    # Run the model
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    # Process outputs
    boxes = outputs['boxes']
    labels = outputs['labels']
    scores = outputs['scores']

    # Scale boxes back to original frame size
    scale_x = frame.shape[1] / 300
    scale_y = frame.shape[0] / 300

    for box, label, score in zip(boxes, labels, scores):
        if score > CONFIDENCE_THRESHOLD:
            # Rescale box
            xmin, ymin, xmax, ymax = box
            xmin = int(xmin * scale_x)
            ymin = int(ymin * scale_y)
            xmax = int(xmax * scale_x)
            ymax = int(ymax * scale_y)

            # Draw rectangle and label on original frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
            text = f'{class_name}: {score:.2f}'
            cv2.putText(frame, text, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
