import time
import cv2
import os
import numpy as np
from glob import glob

# Parameters
input_folder = 'input'
background_subtractor_history = 50
background_subtractor_threshold = 400.0
cluster_identification_threshold = 100
accumulate_frames = 20
movement_frames_needed = 5
proximity_to_merge_boxes = 10
video_padding = 50
warmup_frames = 10

def detect_motion_regions(
    input_path,
    clip_index,
    background_subtractor_history = 50,
    background_subtractor_threshold = 400.0,
    cluster_identification_threshold = 100,
    accumulate_frames = 20,
    movement_frames_needed = 5,
    proximity_to_merge_boxes = 10,
    video_padding = 50,
    warmup_frames = 10,
):
    def merge_close_boxes(boxes, proximity=50):
        merged = True
        while merged:
            merged = False
            new_boxes = []
            skip = set()
            for i in range(len(boxes)):
                if i in skip:
                    continue
                x1, y1, w1, h1 = boxes[i]
                box1 = [x1, y1, x1 + w1, y1 + h1]
                has_merged = False
                for j in range(i + 1, len(boxes)):
                    if j in skip:
                        continue
                    x2, y2, w2, h2 = boxes[j]
                    box2 = [x2, y2, x2 + w2, y2 + h2]
                    if not (box2[0] > box1[2] + proximity or box2[2] < box1[0] - proximity or
                            box2[1] > box1[3] + proximity or box2[3] < box1[1] - proximity):
                        box1[0] = min(box1[0], box2[0])
                        box1[1] = min(box1[1], box2[1])
                        box1[2] = max(box1[2], box2[2])
                        box1[3] = max(box1[3], box2[3])
                        skip.add(j)
                        merged = True
                        has_merged = True
                new_boxes.append((box1[0], box1[1], box1[2] - box1[0], box1[3] - box1[1]))
            boxes = new_boxes
        return boxes

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fgbg = cv2.createBackgroundSubtractorKNN(history=background_subtractor_history, dist2Threshold=background_subtractor_threshold, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    heatmap = None
    frame_count = 0
    boxes2 = []
    frame_index = 0
    metadata = []
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        fgmask = fgbg.apply(blurred)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        frame_index+=1
        if frame_index <= warmup_frames:
            continue
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            merged_mask = np.zeros_like(fgmask)
            cv2.drawContours(merged_mask, contours, -1, 255, -1)
            contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if heatmap is None:
            heatmap = np.zeros_like(fgmask, dtype=np.float32)
        heatmap += fgmask.astype(np.float32)
        frame_count += 1
        if frame_count >= accumulate_frames:
            boxes2 = []
            thresh = ((heatmap >= movement_frames_needed).astype(np.uint8)) * 255
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < cluster_identification_threshold:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                boxes2.append((x, y, w, h))
            boxes2 = merge_close_boxes(boxes2, proximity=proximity_to_merge_boxes)
            if boxes2:
                metadata.append((frame_index - accumulate_frames, frame_index - 1, boxes2))
            heatmap.fill(0)
            frame_count = 0
    end = time.time()
    print(f'Identified regions of interest in {end - start:.2f} seconds')
    cap.release()
    os.makedirs("videos", exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    for (start_f, end_f, boxes) in metadata:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames_to_save = end_f - start_f + 1
        buffer = []
        for _ in range(frames_to_save):
            ret, frame = cap.read()
            if not ret:
                break
            buffer.append(frame.copy())
        for box in boxes:
            x, y, w, h = box
            crop_x1 = max(x - video_padding, 0)
            crop_y1 = max(y - video_padding, 0)
            crop_x2 = min(x + w + video_padding, buffer[0].shape[1])
            crop_y2 = min(y + h + video_padding, buffer[0].shape[0])
            out_path = f"videos/{clip_index[0]}.mp4"
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                                (crop_x2 - crop_x1, crop_y2 - crop_y1))
            for f in buffer:
                crop = f[crop_y1:crop_y2, crop_x1:crop_x2]
                out.write(crop)
            out.release()
            clip_index[0] += 1
    cap.release()
    end = time.time()
    print(f'Finished in {end - start:.2f} seconds')

def process_folder(input_folder):
    video_files = sorted(glob(os.path.join(input_folder, "*.mp4")))
    clip_index = [1]  # Use list for mutable integer
    for video_file in video_files:
        print(f"Processing {video_file} ...")
        detect_motion_regions(
            video_file,
            clip_index,
            background_subtractor_history = background_subtractor_history,
            background_subtractor_threshold = background_subtractor_threshold,
            cluster_identification_threshold = cluster_identification_threshold,
            accumulate_frames = accumulate_frames,
            movement_frames_needed = movement_frames_needed,
            proximity_to_merge_boxes = proximity_to_merge_boxes,
            video_padding = video_padding,
            warmup_frames = warmup_frames,
        )

process_folder(input_folder)