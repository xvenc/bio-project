import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from bob.bio.vein.extractor import RepeatedLineTracking
from bob.bio.vein.preprocessor import FixedMask

IMG = "vein_img/test1.bmp"

# Load the grayscale image
img = np.asarray(Image.open(IMG).convert('L'))

# Create mask of where the finger is - using simple fixed masked for now
fmask = FixedMask(60, 40, 20, 20)
mask = fmask(img)

# Initialize the RepeatedLineTracking algorithm
rtl = RepeatedLineTracking(iterations=1000, rescale=False)
vein_lines = rtl.repeated_line_tracking(img, mask)

# Create a figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display both images side-by-side
axes[0].imshow(img, cmap='gray')
axes[0].axis('off')
axes[1].imshow(vein_lines, cmap='gray')
axes[1].axis('off')

plt.show()

