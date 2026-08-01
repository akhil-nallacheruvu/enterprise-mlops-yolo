import numpy as np
from PIL import Image
import glob

image_paths = glob.glob("data/*.jpg") + glob.glob("data/*.png")

if not image_paths:
    raise ValueError("No images found in data/ — point this at your reference dataset")

brightness_values = []
for path in image_paths:
    img = np.array(Image.open(path).convert("RGB"))
    brightness_values.append(img.mean())

mean_brightness = np.mean(brightness_values)
std_brightness = np.std(brightness_values)

print(f"BASELINE_MEAN_BRIGHTNESS = {mean_brightness:.2f}")
print(f"BASELINE_STD_BRIGHTNESS = {std_brightness:.2f}")
print(f"(computed from {len(image_paths)} reference images)")
