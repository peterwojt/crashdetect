import cv2
import os
import numpy as np
import pandas as pd
from typing import List

# --------- Optical Flow + Feature Extraction ----------

def compute_optical_flow(video_frames: List[np.ndarray]) -> List[np.ndarray]:
    flows = []
    for i in range(len(video_frames) - 1):
        prev = cv2.cvtColor(video_frames[i], cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(video_frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, next, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)
    return flows

def compute_motion_features(video_frames: List[np.ndarray]) -> dict:
    flows = compute_optical_flow(video_frames)
    mag_list, ang_list, motion_area_list = [], [], []

    framewise_mean_mags = []
    framewise_motion_area = []
    motion_threshold = 1.0

    h, w = video_frames[0].shape[:2]
    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_margin = int(min(h, w) * 0.25)
    center_mask[center_margin:-center_margin, center_margin:-center_margin] = 1
    edge_mask = 1 - center_mask

    center_motion = []
    edge_motion = []

    for flow in flows:
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_list.append(mag)
        ang_list.append(ang)

        motion_pixels = mag > motion_threshold
        motion_area_ratio = np.sum(motion_pixels) / (h * w)
        motion_area_list.append(motion_area_ratio)

        framewise_mean_mags.append(np.mean(mag))
        framewise_motion_area.append(motion_area_ratio)

        center_motion.append(np.mean(mag[center_mask == 1]))
        edge_motion.append(np.mean(mag[edge_mask == 1]))

    mag_stack = np.stack(mag_list)
    flat_mag = mag_stack.flatten()
    mean_mag = np.mean(flat_mag)
    std_mag = np.std(flat_mag)
    max_mag = np.max(flat_mag)
    min_mag = np.min(flat_mag[np.nonzero(flat_mag)])
    median_mag = np.median(flat_mag)
    perc_95_mag = np.percentile(flat_mag, 95)

    ang_stack = np.stack(ang_list)
    flat_ang = ang_stack.flatten()
    mean_ang = np.mean(flat_ang)
    std_ang = np.std(flat_ang)

    hist, _ = np.histogram(flat_ang, bins=36, range=(0, 2 * np.pi), density=True)
    direction_entropy = -np.sum(hist * np.log(hist + 1e-7))

    mean_center_motion = np.mean(center_motion)
    mean_edge_motion = np.mean(edge_motion)
    center_edge_ratio = mean_center_motion / (mean_edge_motion + 1e-7)

    motion_peak_count = np.sum(np.array(framewise_mean_mags) > (mean_mag + std_mag))
    motion_std_over_time = np.std(framewise_mean_mags)

    x = np.arange(len(framewise_mean_mags))
    mag_trend = np.polyfit(x, framewise_mean_mags, 1)[0]
    area_trend = np.polyfit(x, framewise_motion_area, 1)[0]

    avg_motion_map = np.mean(mag_stack, axis=0)
    norm_map = avg_motion_map / (np.sum(avg_motion_map) + 1e-7)
    motion_heatmap_entropy = -np.sum(norm_map * np.log(norm_map + 1e-7))

    features = {
        'mean_magnitude': mean_mag,
        'std_magnitude': std_mag,
        'max_magnitude': max_mag,
        'min_magnitude': min_mag,
        'median_magnitude': median_mag,
        'perc_95_magnitude': perc_95_mag,
        'mean_angle': mean_ang,
        'std_angle': std_ang,
        'direction_entropy': direction_entropy,
        'motion_pixel_ratio': np.mean(motion_area_list),
        'max_motion_area': np.max(motion_area_list),
        'motion_area_variance': np.var(motion_area_list),
        'center_vs_edge_motion_ratio': center_edge_ratio,
        'motion_peak_count': motion_peak_count,
        'motion_std_over_time': motion_std_over_time,
        'motion_magnitude_trend': mag_trend,
        'motion_area_trend': area_trend,
        'motion_heatmap_entropy': motion_heatmap_entropy
    }

    return features

# --------- Video Loader ----------

def load_first_n_frames(video_path: str, num_frames: int = 20) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = resize_with_aspect_ratio(frame, width=640)
        frames.append(frame)
    cap.release()
    return frames if len(frames) == num_frames else []

# --------- Main Script ----------

def process_video_folder(video_folder: str, output_csv: str):
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    output_fields = None
    file_exists = os.path.exists(output_csv)

    with open(output_csv, 'a') as out_csv:
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            #print(f"Processing: {video_path}")
            try:
                frames = load_first_n_frames(video_path, num_frames=20)
                if not frames:
                    print(" - Skipped (too short or unreadable)")
                    continue
                features = compute_motion_features(frames)
                features['video_filename'] = video_file

                df = pd.DataFrame([features])
                df.to_csv(out_csv, mode='a', header=not file_exists and output_fields is None, index=False)
                file_exists = True  # Ensure header isn't written again
                #print(" - Success")
            except Exception as e:
                print(f" - Error: {e}")

def resize_with_aspect_ratio(frame, width=None, height=None):
    (h, w) = frame.shape[:2]

    if width is None and height is None:
        return frame  # no resizing

    if width is not None:
        # calculate new height to keep ratio
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        # calculate new width to keep ratio
        r = height / float(h)
        dim = (int(w * r), height)

    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized


# --------- Run this ---------

video_folder = 'car_crash_video_dataset/val/crash'
output_csv = 'motion_info/val_crash.csv'
process_video_folder(video_folder, output_csv)
