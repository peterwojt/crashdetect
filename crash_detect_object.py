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
# Track class with velocity + crash detection
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

        # velocity
        self.vx = 0.0
        self.vy = 0.0
        self.speed_history = []  # stores speeds per frame
        self.crash_flag = False  # rapid deceleration

    def update(self, bbox, score, label, timestamp_frame, alpha=0.6):
        # velocity calculation
        cx_old = (self.bbox[0] + self.bbox[2]) / 2.0
        cy_old = (self.bbox[1] + self.bbox[3]) / 2.0
        cx_new = (bbox[0] + bbox[2]) / 2.0
        cy_new = (bbox[1] + bbox[3]) / 2.0
        new_vx = cx_new - cx_old
        new_vy = cy_new - cy_old
        self.vx = 0.6 * self.vx + 0.4 * new_vx
        self.vy = 0.6 * self.vy + 0.4 * new_vy

        # smooth bbox
        self.bbox = alpha * self.bbox + (1 - alpha) * np.array(bbox, dtype=float)
        self.score = max(self.score, float(score))
        self.label_history.append(label)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_seen = timestamp_frame

        # speed & deceleration
        speed = np.sqrt(self.vx**2 + self.vy**2)
        self.speed_history.append(speed)
        if len(self.speed_history) >= 2:
            last_speed = self.speed_history[-2]
            accel = speed - last_speed
            if accel < -5:  # strong deceleration threshold (tune for your video)
                self.crash_flag = True

    def predict(self):
        # shift bbox based on velocity
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
# Tracker class
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

        # predict future positions
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
                iou_val = iou(track.bbox, db)
                cost_matrix[t_idx, d_idx] = 1 - iou_val

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()

        for t_idx, d_idx in zip(row_ind, col_ind):
            iou_val = 1 - cost_matrix[t_idx, d_idx]
            if iou_val < self.iou_threshold:
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
# Main loop
# -------------------------
if __name__ == "__main__":
    #cap = cv2.VideoCapture('crashes/2023-2024/110_NE_4_-_Center_2024-04-18_20_18_19_042.mp4')
    #cap = cv2.VideoCapture('media_w1117040928_7.ts')
    cap = cv2.VideoCapture('crashes/2023-2024/108_NE_8-_-_Center_2023-12-03_16_19_38_749.mp4')
    tracker = SimpleTracker(iou_threshold=0.1, max_age=15, min_hits=2, sticky_label=False)

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

        for tr in tracks:
            x1, y1, x2, y2, tid, _ = tr.to_output()
            label_idx = tr.get_label(sticky=tracker.sticky_label)
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx] if 0 <= label_idx < len(COCO_INSTANCE_CATEGORY_NAMES) else "N/A"

            color = (0, 255, 0)
            if tr.crash_flag:
                color = (0, 0, 255)  # red for potential crash
                cv2.putText(frame, "CRASH!", (x1, max(y1 - 30, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {tid} {class_name}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, f"Inference: {inference_time_ms:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Motion-Aware Tracker + Crash Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
