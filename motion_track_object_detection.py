import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F
import cv2
import numpy as np
from tracker import Sort  # Make sure this file is in the same folder

# Load model
model = ssd300_vgg16(pretrained=True).eval()

# Class names (COCO)
COCO_CLASSES = [
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

# Initialize tracker
tracker = Sort()

# Confidence threshold
CONF_THRESH = 0.5

#cap = cv2.VideoCapture('car_crash_video_dataset/train/crash/2110.mp4')  # Or 'your_video.mp4'
cap = cv2.VideoCapture('crashes/110_NE_4_-_Center_2024-04-18_20_18_19_042.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    resized = cv2.resize(frame, (300, 300))
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0)

    with torch.no_grad():
        detections = model(img_tensor)[0]

    # Prepare detections for SORT: [x1, y1, x2, y2, score]
    dets = []
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if score > CONF_THRESH:
            x1, y1, x2, y2 = box
            # Scale boxes to original frame size
            scale_x = w / 300
            scale_y = h / 300
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            dets.append([x1.item(), y1.item(), x2.item(), y2.item(), score.item()])

    dets = np.array(dets)

    # Update tracker
    tracked_objects = tracker.update(dets)

    # Draw tracked boxes with IDs
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(obj_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
