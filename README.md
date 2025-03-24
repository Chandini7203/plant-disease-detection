# Plant Leaf Disease Detection

## Overview

This project uses machine learning to detect diseases in plant leaves based on images. The project uses a Random Forest classifier and color histogram features to identify diseases.

## Dependencies

*   numpy
*   opencv-python
*   scikit-learn
*   Pillow
*   imghdr

## Setup

1.  Install dependencies:

    ```
    pip install numpy opencv-python scikit-learn Pillow
    ```

2.  Download the PlantVillage dataset and place it in the specified directory.

3.  Update the data directory path in `feature_extraction.py`.

## Usage

1.  Run `feature_extraction.py` to extract features from the images.

    ```
    python feature_extraction.py
    ```

2.  Run `prepare_data.py` to prepare the training and testing data.

    ```
    python prepare_data.py
    ```

3.  Run `train_model.py` to train the machine learning model.

    ```
    python train_model.py
    ```

4.  Run `app.py` to predict the disease for a given leaf image.

    ```
    python app.py
    ```

    Ensure that `label_encoder_classes` is equal to the correct order. This can be done using the following python script.
    ```
    import os
    DATA_DIR = r'C:\Users\user\Downloads\archive\PlantVillage' # Put in your dataset
    classes = os.listdir(DATA_DIR)
    print(classes)
    ```

## Test your model

1.  Create a folder called `test_images`
2.  Copy your testing images inside this folder. The structure should be in the following format.

test_images/
├── Pepper__bell___Bacterial_spot/
│ ├── image1.jpg
│ ├── image2.jpg
├── Potato___Early_blight/
│ ├── image1.jpg
│ ├── image2.jpg
├── ...


3.  Update the line.

test_data_dir = 'test_images' # Update this to your test images directory


4.  Run `app.py`

## Contributing

Feel free to fork this repository and submit pull requests to contribute to this project.

## License

[Optional: Add a license, e.g., MIT License]