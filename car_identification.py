import time
import cv2
import os
import numpy as np

cap = cv2.VideoCapture('input/t.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)


#fgbg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=10, detectShadows=False)
fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Optional: Scale down video for efficient processing
    #frame = cv2.resize(frame, (640, 360))

    # Turns image to grayscale and applies Gaussian blur
    # Improves background subtraction especially under bad lighting
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Applies background subtraction to frame
    fgmask = fgbg.apply(blurred)

    # Applies morphological filtering
    # Reduces noise from video
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    # Merges nearby areas of movement
    if contours:
        merged_mask = np.zeros_like(fgmask)
        cv2.drawContours(merged_mask, contours, -1, 255, -1)
        contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    for cnt in contours:
        if cv2.contourArea(cnt) < 200:  # filter out small noise
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('Moving Cars', frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if False:
    input_folder = 'input'
    output_folder = 'output'

    filename = 'car_crash_video_8.mp4'

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    end = time.time()

    print(f'Processed file in {end - start:.2f} seconds')

    cap.release()
    out.release()
