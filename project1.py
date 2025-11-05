import cv2
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

# Read first frame
ret, prev_frame = cap.read()
if not ret:
    print("❌ Cannot read from camera")
    exit()

# Convert to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Create HSV mask for visualization
hsv_mask = np.zeros_like(prev_frame)
hsv_mask[..., 1] = 255  # Full saturation

# FPS calculation variables
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow (Farneback method)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # Convert flow vectors to polar coordinates (magnitude & angle)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set hue according to flow direction
    hsv_mask[..., 0] = angle * 180 / np.pi / 2

    # Set value according to flow magnitude (normalized)
    hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR for visualization
    motion_overlay = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    # Blend the motion overlay with live webcam feed
    blended = cv2.addWeighted(frame, 0.7, motion_overlay, 0.7, 0)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS
    cv2.putText(blended, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show output
    cv2.imshow("Live Motion Flow Overlay", blended)

    # Update previous frame
    prev_gray = gray.copy()

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
