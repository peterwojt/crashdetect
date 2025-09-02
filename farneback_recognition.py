import cv2
import numpy as np

farneback_params = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

grid_size = 16
frames_confirm = 3
crash_counter = 0

cap = cv2.VideoCapture("videos/522.mp4")
ret, frame1 = cap.read()
if not ret:
    print("Error: cannot read video")
    exit()

prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
h, w = prev_gray.shape
prev_mag = np.zeros_like(prev_gray, dtype=np.float32)

while True:
    ret, frame2 = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame2 = cap.read()
        if not ret:
            break
        prev_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        continue

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # --- Optical flow ---
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **farneback_params)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    crash_detected = False
    regions = []

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            cell_mag = mag[y:y+grid_size, x:x+grid_size]
            cell_ang = ang[y:y+grid_size, x:x+grid_size]
            prev_cell_mag = prev_mag[y:y+grid_size, x:x+grid_size]

            if cell_mag.size == 0:
                continue

            mean_mag = np.mean(cell_mag)
            std_ang = np.std(cell_ang)
            drop = np.mean(prev_cell_mag) - mean_mag  # sudden slowdown

            # Crash-like condition:
            if mean_mag > 2.5 and std_ang > 1.0 and drop > 1.5:
                regions.append((x, y, grid_size, grid_size))

    # --- Merge small regions into bounding boxes ---
    mask = np.zeros((h, w), np.uint8)
    for (x, y, gw, gh) in regions:
        cv2.rectangle(mask, (x, y), (x+gw, y+gh), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400 or area > 0.2 * (h * w):  # ignore tiny and huge (like trucks)
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(frame2, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
        crash_detected = True

    # --- Persistence filter ---
    if crash_detected:
        crash_counter += 1
    else:
        crash_counter = max(0, crash_counter - 1)

    if crash_counter >= frames_confirm:
        cv2.putText(frame2, "🚨 CRASH DETECTED 🚨", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    # --- Display ---
    scale = min(1280 / w, 720 / h)
    disp = cv2.resize(frame2, (int(w*scale), int(h*scale)))
    cv2.imshow("Crash Detection", disp)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

    prev_gray = gray
    prev_mag = mag.copy()

cap.release()
cv2.destroyAllWindows()
