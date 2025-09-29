import cv2
import torch
import time
import numpy as np
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

# -------------------------
# SETUP
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True).eval().to(device)

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

transform = transforms.Compose([transforms.ToTensor()])
DETECTION_THRESHOLD = 0.6
CRASH_PRED_STEPS = 3  # frames to predict ahead for crash

# -------------------------
# IoU helper
# -------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union

# -------------------------
# Track class with velocity
# -------------------------
class Track:
    def __init__(self, bbox, score, label, track_id, timestamp_frame):
        self.bbox = np.array(bbox, dtype=float)
        self.score = float(score)
        self.label_history = [label]
        self.first_label = label
        self.track_id = track_id
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        self.last_seen = timestamp_frame
        self.vx = 0.0
        self.vy = 0.0

    def update(self, bbox, score, label, timestamp_frame, alpha=0.6):
        cx_old = (self.bbox[0] + self.bbox[2]) / 2.0
        cy_old = (self.bbox[1] + self.bbox[3]) / 2.0
        cx_new = (bbox[0] + bbox[2]) / 2.0
        cy_new = (bbox[1] + bbox[3]) / 2.0
        self.vx = 0.6 * self.vx + 0.4 * (cx_new - cx_old)
        self.vy = 0.6 * self.vy + 0.4 * (cy_new - cy_old)

        self.bbox = alpha * self.bbox + (1 - alpha) * np.array(bbox, dtype=float)
        self.score = max(self.score, float(score))
        self.label_history.append(label)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_seen = timestamp_frame

    def predict(self):
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        cx_pred = cx + self.vx
        cy_pred = cy + self.vy
        self.bbox = np.array([
            cx_pred - w / 2, cy_pred - h / 2,
            cx_pred + w / 2, cy_pred + h / 2
        ], dtype=float)

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1

    def get_label(self, sticky=True, majority_window=10):
        if sticky:
            return self.first_label
        hist = self.label_history[-majority_window:]
        vals, counts = np.unique(hist, return_counts=True)
        return vals[np.argmax(counts)]

    def to_output(self):
        x1, y1, x2, y2 = self.bbox.astype(int)
        return [x1, y1, x2, y2, self.track_id, self.score]

# -------------------------
# Tracker
# -------------------------
class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=1, sticky_label=True):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.sticky_label = sticky_label
        self.tracks = []
        self._next_id = 1
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        for tr in self.tracks:
            tr.predict()

        det_bboxes = [d[:4] for d in detections]
        det_scores = [d[4] for d in detections]
        det_labels = [d[5] for d in detections]

        if len(self.tracks) == 0:
            for bbox, score, label in zip(det_bboxes, det_scores, det_labels):
                self.tracks.append(Track(bbox, score, label, self._next_id, self.frame_count))
                self._next_id += 1
            return [t for t in self.tracks if t.hits >= self.min_hits]

        cost_matrix = np.zeros((len(self.tracks), len(det_bboxes)), dtype=float)
        for t_idx, track in enumerate(self.tracks):
            for d_idx, db in enumerate(det_bboxes):
                cost_matrix[t_idx, d_idx] = 1 - iou(track.bbox, db)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_tracks = set()
        assigned_dets = set()

        for t_idx, d_idx in zip(row_ind, col_ind):
            if 1 - cost_matrix[t_idx, d_idx] < self.iou_threshold:
                continue
            self.tracks[t_idx].update(det_bboxes[d_idx], det_scores[d_idx], det_labels[d_idx], self.frame_count)
            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)

        for tidx, tr in enumerate(self.tracks):
            if tidx not in assigned_tracks:
                tr.mark_missed()

        for didx, (bbox, score, label) in enumerate(zip(det_bboxes, det_scores, det_labels)):
            if didx not in assigned_dets:
                self.tracks.append(Track(bbox, score, label, self._next_id, self.frame_count))
                self._next_id += 1

        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits]

# -------------------------
# Predict future bbox for crash
# -------------------------
def predict_future_bbox(track, steps=CRASH_PRED_STEPS):
    cx = (track.bbox[0] + track.bbox[2]) / 2
    cy = (track.bbox[1] + track.bbox[3]) / 2
    w = track.bbox[2] - track.bbox[0]
    h = track.bbox[3] - track.bbox[1]
    cx_pred = cx + track.vx * steps
    cy_pred = cy + track.vy * steps
    return np.array([cx_pred - w / 2, cy_pred - h / 2, cx_pred + w / 2, cy_pred + h / 2])

# -------------------------
# Main loop
# -------------------------
if __name__ == "__main__":
    #cap = cv2.VideoCapture('crashes/2023-2024/110_NE_4_-_Center_2024-04-18_20_18_19_042.mp4')
    cap = cv2.VideoCapture('media_w1117040928_7.ts')
    #cap = cv2.VideoCapture('crashes/2023-2024/108_NE_8-_-_Center_2023-12-03_16_19_38_749.mp4')
    tracker = SimpleTracker(iou_threshold=0.1, max_age=15, min_hits=3, sticky_label=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image).to(device)

        with torch.no_grad():
            outputs = model([image_tensor])
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000

        pred = outputs[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score < DETECTION_THRESHOLD:
                continue
            x1, y1, x2, y2 = box.astype(int)
            detections.append([x1, y1, x2, y2, float(score), int(label)])

        tracks = tracker.update(detections)

        # Draw tracks
        for tr in tracks:
            x1, y1, x2, y2, tid, _ = tr.to_output()
            label_idx = tr.get_label(sticky=tracker.sticky_label)
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx] if 0 <= label_idx < len(COCO_INSTANCE_CATEGORY_NAMES) else "N/A"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {tid} {class_name}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- Velocity-based crash detection ---
        car_tracks = [tr for tr in tracks if COCO_INSTANCE_CATEGORY_NAMES[tr.get_label()].lower() == "car"]
        for i in range(len(car_tracks)):
            for j in range(i + 1, len(car_tracks)):
                tr1, tr2 = car_tracks[i], car_tracks[j]
                future1, future2 = predict_future_bbox(tr1), predict_future_bbox(tr2)
                if iou(future1, future2) > 0.05:  # small overlap indicates potential crash
                    x1_1, y1_1, x2_1, y2_1 = future1.astype(int)
                    x1_2, y1_2, x2_2, y2_2 = future2.astype(int)
                    cv2.rectangle(frame, (x1_1, y1_1), (x2_1, y2_1), (0, 0, 255), 3)
                    cv2.rectangle(frame, (x1_2, y1_2), (x2_2, y2_2), (0, 0, 255), 3)
                    cv2.putText(frame, "!!! POTENTIAL CRASH !!!", (min(x1_1, x1_2), max(y1_1, y1_2) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        cv2.putText(frame, f"Inference: {inference_time_ms:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Motion-Aware Tracker with Crash Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
