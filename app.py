import os
import cv2
import numpy as np
import joblib
import os

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Load trained model and label encoder
model = joblib.load('random_forest_model.pkl')

# Assuming these are all the classes in your dataset
label_encoder_classes = ['Pepper__bell___Bacterial_spot', 'Potato___Early_blight', 'Potato___Late_blight', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus']

def predict_disease(image_path):
    feature_vector = extract_features(image_path)
    prediction = model.predict([feature_vector])
    predicted_class_index = prediction[0]
    return label_encoder_classes[predicted_class_index]

# Directory containing test images
test_data_dir = 'test_images'  # Update this to your test images directory

# Loop through each class directory
for class_name in os.listdir(test_data_dir):
    class_dir = os.path.join(test_data_dir, class_name)

    # Loop through each image in the class directory
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)

        # Predict the disease for the current image
        predicted_disease = predict_disease(img_path)
        print(f"Image: {img_path}, Predicted Disease: {predicted_disease}")
