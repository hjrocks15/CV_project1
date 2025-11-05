import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("❌ Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=100, param2=40, minRadius=10, maxRadius=200
    )

    _, binary = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ---- Circles ----
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(frame, "Circle", (x - 20, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ---- Ellipses (fixed) ----
    for cnt in contours:
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            if MA <= 0 or ma <= 0:
                continue

            cv2.ellipse(frame, ellipse, (255, 0, 0), 2)
            cv2.putText(frame, "Ellipse", (int(x - 30), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Smart Shape Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
