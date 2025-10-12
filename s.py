import os
import csv
import cv2

def save_crash_clips(csv_file="crashes_in_videos.csv", 
                     output_folder="cropped_crash_videos", 
                     pre_frames=8, post_frames=7):
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
            # Compute ideal start/end
            start_f = crash_frame - pre_frames
            end_f = crash_frame + post_frames

            # Adjust if near start or end
            if start_f < 0:
                # Shift forward to keep range valid
                end_f += abs(start_f)
                start_f = 0
            if end_f >= total_frames:
                # Shift backward if we exceed total frames
                shift_back = end_f - (total_frames - 1)
                start_f = max(0, start_f - shift_back)
                end_f = total_frames - 1

            # Ensure final range gives exactly 16 frames (if possible)
            expected_frames = pre_frames + post_frames + 1
            actual_frames = end_f - start_f + 1


            width = x2 - x1
            height = y2 - y1

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            frames = []
            for f in range(start_f, end_f + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if not ret:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                frames.append(crop)

            # 🧩 Pad if too few frames (duplicate first/last as needed)
            while len(frames) < expected_frames:
                if len(frames) == 0:
                    break
                if start_f == 0:
                    frames.insert(0, frames[0].copy())  # pad with first frame
                else:
                    frames.append(frames[-1].copy())    # pad with last frame

            # Write all frames to output video
            for crop in frames:
                out.write(crop)

            out.release()
            cap.release()

            print(f"✅ Saved {out_path} ({end_f - start_f + 1} frames)")

    print(f"\n📁 All cropped crash videos saved in: {output_folder}")

if __name__ == "__main__":
    save_crash_clips("crashes_in_videos.csv")
