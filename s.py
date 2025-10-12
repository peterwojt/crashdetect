import os
import csv
import cv2
import numpy as np

def save_crash_clips(csv_file="crashes_in_videos.csv", 
                     output_folder="cropped_crash_videos2", 
                     pre_frames=7, post_frames=8):
    os.makedirs(output_folder, exist_ok=True)

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = row["video_file"]
            crash_frame = int(row["frame"])
            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])

            base_name = os.path.splitext(os.path.basename(video_path))[0]
            out_name = f"{base_name}_crash_{crash_frame}.mp4"
            out_path = os.path.join(output_folder, out_name)

            print(f"🎬 Processing {out_name} ...")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️ Could not open {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            start_f = crash_frame - pre_frames
            end_f = crash_frame + post_frames
            desired_frames = end_f - start_f + 1  # should be 16

            width = x2 - x1
            height = y2 - y1
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            for f in range(start_f, end_f + 1):
                if f < 0 or f >= total_frames:
                    # out of bounds → pad with black frame
                    black = np.zeros((height, width, 3), dtype=np.uint8)
                    out.write(black)
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if not ret:
                    # failed read → also pad
                    black = np.zeros((height, width, 3), dtype=np.uint8)
                    out.write(black)
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((height, width, 3), dtype=np.uint8)
                out.write(crop)

            out.release()
            cap.release()

            print(f"✅ Saved {out_path} ({desired_frames} frames padded to 16)")

    print(f"\n📁 All cropped crash videos saved in: {output_folder}")


if __name__ == "__main__":
    save_crash_clips("crashes_in_videos.csv")
