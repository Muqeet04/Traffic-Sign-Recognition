# German Traffic Sign Recognition (CNN)

## Overview
A Deep Learning project that classifies images of traffic signs into 43 distinct categories using a Convolutional Neural Network (CNN). This model is trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

## Dataset
- **Source:** [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Classes:** 43 types of traffic signs (e.g., Stop, Speed Limit, Yield).
- **Structure:** `Train` folder (images sorted by class) and `Test.csv` (metadata for testing).

## Model Architecture
- **Framework:** TensorFlow / Keras
- **Type:** Sequential CNN
- **Layers:**
  - 2x Conv2D (32 filters) + MaxPool + Dropout
  - 2x Conv2D (64 filters) + MaxPool + Dropout
  - Dense (Fully Connected) Layer (256 units)
  - Output Layer (Softmax activation, 43 units)

## Performance
- **Training Accuracy:** ~95%+
- **Test Set Accuracy:** Evaluated on unseen data from `Test.csv`.

## Technologies Used
- TensorFlow & Keras
- OpenCV / PIL (Image Processing)
- Scikit-Learn (Train/Test Split)
- Matplotlib (Performance plotting)

## Usage
1. Run the notebook to download the dataset via `kagglehub`.
2. The model trains for 10 epochs and saves as `traffic_classifier.h5`.
3. The script outputs a final accuracy score on the test set.
