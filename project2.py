import cv2
import numpy as np
from skimage.morphology import skeletonize
from PIL import Image, ImageTk
import tkinter as tk

# --- Webcam + Processing Setup ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("❌ Webcam not found. Please check your camera connection.")

# --- Tkinter Window Setup ---
root = tk.Tk()
root.title("Webcam Skeletonization")

# Labels to show the binary & skeleton images
lbl_binary = tk.Label(root)
lbl_binary.grid(row=0, column=0, padx=10, pady=10)

lbl_skeleton = tk.Label(root)
lbl_skeleton.grid(row=0, column=1, padx=10, pady=10)

def process_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, process_frame)
        return

    # Resize for speed
    frame = cv2.resize(frame, (320, 240))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Skeletonization expects binary in 0-1 range
    skeleton = skeletonize(binary // 255)

    # Convert back to 8-bit
    skeleton_display = (skeleton * 255).astype(np.uint8)

    # Convert for Tkinter (Binary)
    img_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    img_binary = Image.fromarray(img_binary)
    imgtk_binary = ImageTk.PhotoImage(image=img_binary)
    lbl_binary.imgtk = imgtk_binary
    lbl_binary.configure(image=imgtk_binary, text="Binary")

    # Convert for Tkinter (Skeleton)
    img_skeleton = cv2.cvtColor(skeleton_display, cv2.COLOR_GRAY2RGB)
    img_skeleton = Image.fromarray(img_skeleton)
    imgtk_skeleton = ImageTk.PhotoImage(image=img_skeleton)
    lbl_skeleton.imgtk = imgtk_skeleton
    lbl_skeleton.configure(image=imgtk_skeleton, text="Skeleton")

    # Call again after short delay
    root.after(10, process_frame)

# --- Start Processing ---
process_frame()

# --- Tkinter Loop ---
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
root.mainloop()

