import cv2
import os
import numpy as np
from PIL import Image

def extract_features(image_path):
    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    # Check if the image is valid before resizing
    if img.size > 0:
        img = cv2.resize(img, (128, 128))
    else:
        print(f"Error: Image has no size {image_path}")
        return None

    # Calculate color histogram
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def is_valid_image(image_path):
    try:
        Image.open(image_path).verify()
        return True
    except (IOError, SyntaxError):
        return False

# Example usage:
data_dir = r'C:\Users\user\Downloads\archive\PlantVillage'
features = []
labels = []

for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)

        # Check if the file exists
        if not os.path.exists(img_path):
            print(f"Error: File not found at {img_path}")
            continue

        # Check if the file is a valid image
        if not is_valid_image(img_path):
            print(f"Warning: Skipping non-image file {img_path}")
            continue

        feature_vector = extract_features(img_path)

        if feature_vector is not None:
            features.append(feature_vector)
            labels.append(class_name)

# Save features and labels for further processing
np.save('features.npy', features)
np.save('labels.npy', labels)
