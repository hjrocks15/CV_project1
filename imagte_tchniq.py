import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

# -------- Load image --------
# Bas yaha apni image ka path de
img_path = "D:\Comp Vision Projects\istockphoto-878155798-612x612.jpg"   # Example: same folder me ek "test.jpg" image rakh do
img = cv2.imread(img_path)

if img is None:
    print("❌ Image not found! Place test.jpg in the same folder.")
    exit()

# Resize image for GUI display
img = cv2.resize(img, (400, 300))

# -------- Image processing techniques --------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

blur = cv2.GaussianBlur(img, (11, 11), 0)
edges = cv2.Canny(img, 100, 200)
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# -------- Tkinter GUI --------
root = Tk()
root.title("Basic Image Processing Techniques")

# Convert OpenCV -> ImageTk for display
def convert(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(cv_img)
    return ImageTk.PhotoImage(img_pil)

# Original
orig_img = convert(img)
gray_img = convert(gray_rgb)
blur_img = convert(blur)
edge_img = convert(edges_rgb)

# -------- Layout --------
Label(root, text="Original Image").grid(row=0, column=0, pady=10)
Label(root, text="Grayscale").grid(row=0, column=1, pady=10)
Label(root, text="Gaussian Blur").grid(row=2, column=0, pady=10)
Label(root, text="Edge Detection").grid(row=2, column=1, pady=10)

Label(root, image=orig_img).grid(row=1, column=0, padx=10)
Label(root, image=gray_img).grid(row=1, column=1, padx=10)
Label(root, image=blur_img).grid(row=3, column=0, padx=10)
Label(root, image=edge_img).grid(row=3, column=1, padx=10)

root.mainloop()
