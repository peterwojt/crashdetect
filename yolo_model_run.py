import cv2
import tensorflow as tf
import numpy as np
import keras_cv

# Load pretrained YOLOv8 model (Pascal VOC 20 classes)
model = keras_cv.models.YOLOV8Detector.from_preset("yolo_v8_m_pascalvoc")

# Open video file or webcam (0 for webcam)
video_path = "media_w1117040928_7.ts"  # or 0 for webcam
cap = cv2.VideoCapture(video_path)

# Class names for Pascal VOC (20 classes)
pascal_voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x - w/2, y - h/2, x + w/2, y + h/2]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to model input size (512x512)
    resized_frame = cv2.resize(rgb_frame, (512, 512))
    # Normalize to [0, 1]
    input_frame = resized_frame / 255.0
    # Add batch dimension
    input_tensor = tf.expand_dims(tf.convert_to_tensor(input_frame, dtype=tf.float32), axis=0)


    # Run inference
    preds = model.predict(input_tensor)
    print(type(preds))
    print(preds)


    boxes = preds["boxes"][0]
    scores = preds["confidence"][0]
    classes = preds["classes"][0]

    # Convert boxes to xyxy and scale to original frame size
    boxes_xyxy = np.array([xywh_to_xyxy(box) for box in boxes])
    height, width, _ = frame.shape
    boxes_scaled = boxes_xyxy * [width, height, width, height]
    boxes_scaled = boxes_scaled.astype(int)

    # Draw boxes on the original frame
    for box, score, cls in zip(boxes_scaled, scores, classes):
        if score < 0.3:
            continue
        x1, y1, x2, y2 = box
        label = pascal_voc_classes[int(cls)]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("YOLOv8 PascalVOC Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
