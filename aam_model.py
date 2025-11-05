import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import exposure
from mpl_toolkits.mplot3d import Axes3D

# Load example 3D MRI (skimage provides a sample one)
from skimage.io import imread
from skimage.color import rgb2gray

mri = rgb2gray(imread("D:/Comp Vision Projects/brain_tumor_dataset/yes/Y2.jpg"))


  # works offline, no need to download anything
  # shape (256, 256)
# Simulate a 3D stack by repeating (for demo)
mri_3d = np.stack([mri]*30, axis=2)

# Enhance contrast for better viewing
mri_3d = exposure.equalize_hist(mri_3d)

# Display few slices
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for i, ax in enumerate(axes):
    ax.imshow(mri_3d[:, :, i*10], cmap='gray')
    ax.set_title(f"Slice {i*10}")
    ax.axis('off')
plt.show()

# Now 3D visualization (simple surface rendering)
x, y = np.mgrid[0:mri_3d.shape[0], 0:mri_3d.shape[1]]
z = mri_3d[:, :, 15]  # take one mid slice as height data

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
ax.set_title("3D Visualization of Brain MRI Slice")
plt.show()

# (Optional) Simulated Active Appearance Model fitting visualization
# For viva: just show how keypoints might be extracted.
from skimage.feature import canny

edges = canny(mri_3d[:, :, 15], sigma=1.5)
plt.imshow(edges, cmap='gray')
plt.title("AAM step: Extracted edges for shape fitting")
plt.axis('off')
plt.show()
