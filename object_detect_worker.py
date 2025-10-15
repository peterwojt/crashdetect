import cv2
import torch
import time
import os
import csv
import numpy as np
from pathlib import Path
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
from datetime import datetime
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
DETECTION_THRESHOLD = 0.65
PREDICTION_FRAMES = 3  # how many frames ahead to predict for collision

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
# Track class
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
        self.ax = 0.0
        self.ay = 0.0
        self.history = []  # stores (frame, cx, cy, vx, vy, speed, ax, ay)


        

    def update(self, bbox, score, label, timestamp_frame, alpha=0.6):
        cx_old = (self.bbox[0] + self.bbox[2]) / 2.0
        cy_old = (self.bbox[1] + self.bbox[3]) / 2.0
        cx_new = (bbox[0] + bbox[2]) / 2.0
        cy_new = (bbox[1] + bbox[3]) / 2.0
        new_vx = 0.6 * self.vx + 0.4 * (cx_new - cx_old)
        new_vy = 0.6 * self.vy + 0.4 * (cy_new - cy_old)
        self.ax = new_vx - self.vx
        self.ay = new_vy - self.vy
        self.vx, self.vy = new_vx, new_vy

        self.bbox = alpha * self.bbox + (1 - alpha) * np.array(bbox, dtype=float)
        self.score = max(self.score, float(score))
        self.label_history.append(label)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_seen = timestamp_frame
        
        speed = np.hypot(self.vx, self.vy)
        self.history.append((timestamp_frame, cx_new, cy_new, self.vx, self.vy, speed, self.ax, self.ay))


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
# Simple tracker
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
# Predict future position
# -------------------------
def predict_future_bbox(track, steps=PREDICTION_FRAMES):
    return track.bbox + np.array([track.vx*steps, track.vy*steps, track.vx*steps, track.vy*steps])

def is_moving(track, min_speed=5.0, min_frames=5):
    """Return True if track has been moving consistently."""
    speed = np.hypot(track.vx, track.vy)
    return speed > min_speed and track.hits > min_frames

def analyze_post_crash(tracker, verified_crashes, post_frames=5, decel_threshold=1.0):
    """
    Check for deceleration after a verified crash.
    - post_frames: number of frames after crash to check
    - decel_threshold: minimum speed drop (pixels/frame) considered deceleration
    """
    crash_results = []

    for c in verified_crashes:
        tr_id1, tr_id2 = c["tracks"]
        tr1 = next((t for t in tracker.tracks if t.track_id == tr_id1), None)
        tr2 = next((t for t in tracker.tracks if t.track_id == tr_id2), None)
        if tr1 is None or tr2 is None:
            continue

        frame_of_crash = c["frame"]
        decel_flag = False

        for tr in [tr1, tr2]:
            # get history entries after the crash frame
            post_hist = [h for h in tr.history if h[0] > frame_of_crash]
            post_hist = post_hist[:post_frames]  # only next few frames

            if len(post_hist) < 2:
                continue  # not enough frames to measure deceleration

            # compute speed drop
            speed_before = next(h for h in tr.history if h[0] == frame_of_crash)[5]
            speed_after = post_hist[-1][5]
            decel = speed_before - speed_after

            if decel >= decel_threshold:
                decel_flag = True

        c["deceleration"] = decel_flag
        crash_results.append(c)

    return crash_results

track_speed_history = {}
post_crash_monitor = []       # monitor deceleration after crash
PRE_CRASH_FRAMES = 5      # frames to average before crash
POST_CRASH_FRAMES = 5     # frames to average after crash
DECEL_PERCENT_THRESHOLD = 30        # speed drop threshold to confirm crash

# -------------------------
# Main loop
# -------------------------

output_csv = "traffic_cam_videos/crash_log.csv"
write_header = not os.path.exists(output_csv)

if write_header:
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_file", 
            "frame",
            "x1",
            "y1",
            "x2",
            "y2"
        ])


roi_corners = np.array([[152, 101], [491, 99], [635, 187],[635,355],[145, 355]], dtype=np.int32)

def process_chunk(filename):
    #cap = cv2.VideoCapture('crashes/156_NE_8_-_E_2024-08-07_13_52_59_610.mp4')
    #cap = cv2.VideoCapture('crashes/Bel-Way_NE_2_-_S_2024-09-30_20_46_57_395.mp4')
    #cap = cv2.VideoCapture('../../Downloads/media_w720815558_5615.ts')
    full_path = os.path.join('traffic_cam_videos/processed', filename)
    cap = cv2.VideoCapture(full_path)
    #cap = cv2.VideoCapture('crashes/110_NE_4_-_Center_2024-04-18_20_18_19_042.mp4')
    #cap = cv2.VideoCapture('media_w1117040928_7.ts')
    #cap = cv2.VideoCapture('crashes/Lk_Hills_Conn_SE_7-8-_-_W_2024-03-28_15_05_49_904.mp4')
    #cap = cv2.VideoCapture('crashes/112_NE_2_-_W_2024-07-18_10_49_00_915.mp4')
    #cap = cv2.VideoCapture('crashes/156_NE_8_-_N_2024-08-07_13_52_59_610.mp4')
    tracker = SimpleTracker(iou_threshold=0.1, max_age=15, min_hits=3, sticky_label=False)
    
    crash_confirmed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        # Create a mask of zeros (black)
        frame_height, frame_width = frame.shape[:2]
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # Fill the polygon area with 1 (white)
        cv2.fillPoly(mask, [roi_corners], 1)

        # Apply mask: black outside, original color inside

        #cv2.imshow("normal", frame)
        #masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        #cv2.imshow("masked", masked_frame)


        frame = cv2.bitwise_and(frame, frame, mask=mask)
        x, y, w, h = cv2.boundingRect(roi_corners)  # rectangle that tightly encloses polygon
        frame = frame[y:y+h, x:x+w]

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

        
        MIN_BOX_AREA = 2000 

        MAX_BOX_AREA = 10000  # ignore overly large boxes (likely false detections)
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score < DETECTION_THRESHOLD:
                continue
            x1, y1, x2, y2 = box.astype(int)
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue  # ignore small detections
            if area > MAX_BOX_AREA:
                continue  # ignore small detections
            detections.append([x1, y1, x2, y2, float(score), int(label)])

        tracks = tracker.update(detections)
        # Make sure every track has a history entry for this frame
        for tr in tracks:
            speed = np.hypot(tr.vx, tr.vy)
            if tr.track_id not in track_speed_history:
                track_speed_history[tr.track_id] = []
            track_speed_history[tr.track_id].append(speed)
            # Keep only the last PRE_CRASH_FRAMES frames
            if len(track_speed_history[tr.track_id]) > PRE_CRASH_FRAMES:
                track_speed_history[tr.track_id].pop(0)

        # Draw tracks and collect cars
        car_tracks = []
        for tr in tracks:
            x1, y1, x2, y2, tid, _ = tr.to_output()
            label_idx = tr.get_label(sticky=tracker.sticky_label)
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx] if 0 <= label_idx < len(COCO_INSTANCE_CATEGORY_NAMES) else "N/A"
            color = (0, 255, 0)
            
            if class_name.lower() in {"car", "truck" ,"motorcycle","bus"}:
                car_tracks.append(tr)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                #cv2.putText(frame, f"ID {tid} {class_name}", (x1, max(y1 - 10, 0)),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # --- Crash detection using future positions ---
        for i in range(len(car_tracks)):
            for j in range(i + 1, len(car_tracks)):
                tr1 = car_tracks[i]
                tr2 = car_tracks[j]

                # IMPORTANT: and means both have to be moving, or means only one
                if not (is_moving(tr1) or is_moving(tr2)):
                    continue

                future1 = predict_future_bbox(tr1)
                future2 = predict_future_bbox(tr2)

                if iou(future1, future2) > 0.1:
                    # Relative velocity
                    v_rel = np.array([tr2.vx - tr1.vx, tr2.vy - tr1.vy])
                    rel_speed = np.linalg.norm(v_rel)

                    # Require meaningful approach speed
                    if rel_speed < 3:  
                        continue  # moving too slowly → likely not a crash

                    # Check angle of approach
                    v1 = np.array([tr1.vx, tr1.vy])
                    v2 = np.array([tr2.vx, tr2.vy])

                    # IMPORTANT: 'or' says one is stopped, 'and' says both are stopped
                    if np.linalg.norm(v1) < 1 or np.linalg.norm(v2) < 1:
                        continue  # one is basically stopped

                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

                    if cos_theta > 0.5:
                        continue  # ~same direction (parallel lanes), skip

                    # Relative position dot velocity → moving towards each other
                    dx = (future2[0]+future2[2])/2 - (future1[0]+future1[2])/2
                    dy = (future2[1]+future2[3])/2 - (future1[1]+future1[3])/2
                    rel_v = v_rel[0]*dx + v_rel[1]*dy

                    if rel_v < 0:
                        # FLAG CRASH
                        x1_1, y1_1, x2_1, y2_1 = tr1.bbox.astype(int)
                        x1_2, y1_2, x2_2, y2_2 = tr2.bbox.astype(int)
                        # Only if this pair is not already monitored
                        existing_monitor = next((m for m in post_crash_monitor if set(m["tracks"]) == {tr1.track_id, tr2.track_id}), None)
                        if existing_monitor is None:
                            post_crash_monitor.append({
                                "tracks": (tr1.track_id, tr2.track_id),
                                "frame": tracker.frame_count,
                                "frames_left": POST_CRASH_FRAMES,
                                "pre_crash_speed_history": {
                                    tr1.track_id: list(track_speed_history.get(tr1.track_id, [np.hypot(tr1.vx, tr1.vy)])),
                                    tr2.track_id: list(track_speed_history.get(tr2.track_id, [np.hypot(tr2.vx, tr2.vy)]))
                                },
                                "post_crash_speed_history": {
                                    tr1.track_id: [],
                                    tr2.track_id: []
                                },
                                'bbox1': tr1.bbox.copy(),       # bbox of first car at detection
                                'bbox2': tr2.bbox.copy(),       # bbox of second car at detection
                                "vx_history": {tr1.track_id: [], tr2.track_id: []},
                                "vy_history": {tr1.track_id: [], tr2.track_id: []},
                                "ax_history": {tr1.track_id: [], tr2.track_id: []},
                                "ay_history": {tr1.track_id: [], tr2.track_id: []},
                                "crash_confirmed": False
                            })

                        #cv2.rectangle(frame, (x1_1, y1_1), (x2_1, y2_1), (0, 0, 255), 3)
                        #cv2.rectangle(frame, (x1_2, y1_2), (x2_2, y2_2), (0, 0, 255), 3)
                        #cv2.putText(frame, "!!! CRASH !!!", (min(x1_1, x1_2), max(y1_1, y1_2) - 10),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        for monitor in post_crash_monitor:
            if monitor["crash_confirmed"] or monitor["frames_left"] <= 0:
                continue

            for tid in monitor["tracks"]:
                tr = next((t for t in tracker.tracks if t.track_id == tid), None)
                if tr is None:
                    continue

                # Speed
                speed = np.hypot(tr.vx, tr.vy)
                monitor["post_crash_speed_history"][tid].append(speed)

                # Velocity
                monitor["vx_history"][tid].append(tr.vx)
                monitor["vy_history"][tid].append(tr.vy)

                # Acceleration (frame-to-frame)
                if len(monitor["vx_history"][tid]) > 1:
                    ax = tr.vx - monitor["vx_history"][tid][-2]
                    ay = tr.vy - monitor["vy_history"][tid][-2]
                else:
                    ax = ay = 0
                monitor["ax_history"][tid].append(ax)
                monitor["ay_history"][tid].append(ay)

            monitor["frames_left"] -= 1


        for monitor in post_crash_monitor:
            if monitor["crash_confirmed"] or monitor["frames_left"] > 0:
                continue

            decel_flags = []
            for tid in monitor["tracks"]:
                pre_speeds = monitor["pre_crash_speed_history"][tid]
                post_speeds = monitor["post_crash_speed_history"][tid]
                if len(post_speeds) == 0:
                    continue

                avg_pre = sum(pre_speeds) / len(pre_speeds)
                avg_post = sum(post_speeds) / len(post_speeds)
                if avg_pre > 0:
                    percent_decel = (avg_pre - avg_post) / avg_pre * 100
                else:
                    percent_decel = 0

                decel_flags.append(percent_decel >= DECEL_PERCENT_THRESHOLD)  # define this threshold, e.g., 20 for 20%

            if any(decel_flags):
                monitor["crash_confirmed"] = True
                crash_confirmed = True

                # Get saved detection bboxes
                x1_1, y1_1, x2_1, y2_1 = monitor['bbox1'].astype(int)
                x1_2, y1_2, x2_2, y2_2 = monitor['bbox2'].astype(int)

                # Frame dimensions and padding
                frame_height, frame_width = frame.shape[:2]
                pad = 50

                # Compute combined box
                comb_x1 = max(0, min(x1_1, x1_2) - pad)
                comb_y1 = max(0, min(y1_1, y1_2) - pad)
                comb_x2 = min(frame_width - 1, max(x2_1, x2_2) + pad)
                comb_y2 = min(frame_height - 1, max(y2_1, y2_2) + pad)

                with open(output_csv, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        video_path,
                        monitor['frame'],
                        comb_x1,
                        comb_y1,
                        comb_x2,
                        comb_y2
                    ])
                #print(f"REAL CRASH confirmed: Tracks {monitor['tracks']} at frame {monitor['frame']} pre: {avg_pre} post: {avg_post} decel: {percent_decel}")

        #cv2.putText(frame, f"Inference: {inference_time_ms:.1f} ms", (10, 30),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        #cv2.imshow("Predictive Crash Tracker", frame)
        
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    cap.release()
    cv2.destroyAllWindows()

    if crash_confirmed:
        crash_path = os.path.join('traffic_cam_videos/crash', filename)
        source = Path(full_path)
        destination = Path(crash_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    else:
        if os.path.exists(full_path):
            os.remove(full_path)

PROCESSED_FOLDER = Path("traffic_cam_videos/processed")
CHECK_INTERVAL = 5 
STOP_TIME = datetime(2025, 10, 21, 3, 0, 0)  # <-- change this to your cutoff (year, month, day, hour, minute, second)

def worker_loop():
    while True:
        if datetime.now() >= STOP_TIME:
            print(f"Stop time reached ({STOP_TIME}), exiting worker loop.")
            break

        files = [f for f in PROCESSED_FOLDER.iterdir() if f.is_file()]
        
        if files:
            for file_path in files:
                process_chunk(file_path.name)
        else:
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    worker_loop()