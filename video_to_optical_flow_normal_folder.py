import os
import cv2
import numpy as np

# --- Configuration ---
input_root = 'car_crash_video_dataset'
output_root = 'optical_flow_dataset2'

# --- Create output folders (same as original layout) ---
for category in ['crashh', 'non_crash']:
    for subset in ['val', 'test'] if category == 'crash' else ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_root, category, subset), exist_ok=True)

# --- Optical Flow Extraction Function ---
def video_to_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    flows = []

    ret, prev = cap.read()
    if not ret:
        print(f"Could not read video: {video_path}")
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        flows.append(flow)
        prev_gray = next_gray

    cap.release()
    return np.array(flows)


# --- Process videos ---
for subset in ['train', 'val', 'test']:
    for category in ['crashh', 'non_crash']:
        input_folder = os.path.join(input_root, subset, category)

        # Skip if the folder doesn’t exist (e.g., crash/train might not exist)
        if not os.path.exists(input_folder):
            continue

        # Determine output folder based on your original layout
        output_folder = os.path.join(output_root, category, subset)
        os.makedirs(output_folder, exist_ok=True)

        print(f"Processing {subset}/{category}...")

        for filename in os.listdir(input_folder):
            if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            video_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.npy'
            output_path = os.path.join(output_folder, output_filename)

            if os.path.exists(output_path):
                continue  # Skip if already processed

            flow = video_to_optical_flow(video_path)
            if flow is not None:
                np.save(output_path, flow)
            else:
                print(f"Failed: {video_path}")

print("✅ Optical flow extraction complete.")
