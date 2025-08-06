import time
import cv2
import os

cap = cv2.VideoCapture('input/car_crash_video_8.mp4')
#fgbg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=10, detectShadows=False)
fgbg = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # filter out small noise
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('Moving Cars', fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
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