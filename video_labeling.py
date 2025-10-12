import cv2
import os
import csv

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

video_folder = 'cropped_crash_videos'  # Replace with your folder path
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
video_files.sort(key=natural_sort_key)
current_index = 0

def save_label(filename, label):
    clean_filename = filename.replace(' ', '')
    labels = {}
    rows = []

    # Read existing labels, skip header
    if os.path.exists('labels2.csv'):
        with open('labels2.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            for row in reader:
                if len(row) == 2:
                    labels[row[0].strip()] = row[1].strip()

    # Update or add label
    labels[clean_filename] = label

    # Write all labels back, no repeats, header only once
    with open('labels2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        for fname, lbl in labels.items():
            writer.writerow([fname, lbl])

def get_label_for_file(filename):
    clean_filename = filename.replace(' ', '')
    if os.path.exists('labels2.csv'):
        with open('labels2.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) == 2 and row[0].strip() == clean_filename:
                    return row[1].strip()
    return None


def play_video(video_path, index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 25

    cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        # Display index in top left corner        # Display index in top left corner with proportional size
        text = f'{index+1}/{len(video_files)}'
        frame_height = frame.shape[0]
        desired_text_height = int(frame_height * 0.05)  # 5% of frame height
        # Estimate font scale for desired height
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        font_scale = desired_text_height / h
        #cv2.putText(frame, text, (10, desired_text_height + 10),
        #            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), 2, cv2.LINE_AA)
        # Display label status in color
        label = get_label_for_file(video_files[index])
        if label == 'n':
            label_text = 'No Crash'
            label_color = (0, 255, 0)  # Green
        elif label == 'c':
            label_text = 'Crash'
            label_color = (0, 0, 255)    # Red
        elif label == 'z':
            label_text = 'Not a car'
            label_color = (0, 255, 255)    # Yellow
        elif label == 'd':
            label_text = 'not included'
            label_color = (255, 255, 255)    # Yellow
        else:
            label_text = ''
            label_color = (200, 200, 200)  # Gray

        cv2.putText(frame, label_text, (10, desired_text_height + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, 2, cv2.LINE_AA)

        
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Video', frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('n'):
            save_label(video_files[index], 'n')
            cap.release()
            cv2.destroyAllWindows()
            return 'next'
        elif key == ord('c'):
            save_label(video_files[index], 'c')
            cap.release()
            cv2.destroyAllWindows()
            return 'next'
        elif key == ord('z'):
            save_label(video_files[index], 'z')
            cap.release()
            cv2.destroyAllWindows()
            return 'next'
        elif key == ord('d'):
            save_label(video_files[index], 'd')
            cap.release()
            cv2.destroyAllWindows()
            return 'next'
        elif key == 83:  # Right arrow key
            # Only allow next if label exists
            if label in ('n', 'c', 'z', 'd'):
                cap.release()
                cv2.destroyAllWindows()
                return 'next'
            # Otherwise ignore right arrow
        elif key == 81:
            cap.release()
            cv2.destroyAllWindows()
            return 'back'
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

# ...existing code...

# Helper to get set of labeled filenames (without spaces)
def get_labeled_files():
    labeled = set()
    if os.path.exists('labels2.csv'):
        with open('labels2.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) == 2:
                    labeled.add(row[0].strip())
    return labeled

# Find first unlabeled file index
labeled_files = get_labeled_files()
start_index = 0
for i, fname in enumerate(video_files):
    clean_fname = fname.replace(' ', '')
    if clean_fname not in labeled_files:
        start_index = i
        break
    # If all are labeled, start at the end
    start_index = len(video_files)

current_index = start_index

# ...existing code...
while 0 <= current_index < len(video_files):
    video_path = os.path.join(video_folder, video_files[current_index])
    action = play_video(video_path, current_index)
    if action == 'next':
        current_index += 1
    elif action == 'back':
        current_index -= 1
    else:
        current_index += 1  # default to next