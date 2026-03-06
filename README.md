# 🚦 Traffic Sign Recognition System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

An end-to-end Deep Learning application that classifies 43 categories of traffic signs using a Convolutional Neural Network (CNN). This project features a web-based dashboard for real-time image inference.

## 🚀 Project Overview
This project addresses the challenge of autonomous vehicle perception by identifying traffic signs from the **German Traffic Sign Recognition Benchmark (GTSRB)**. I moved beyond a simple Jupyter Notebook to build a deployable application that handles image preprocessing and model prediction in a unified pipeline.



[Image of a Convolutional Neural Network architecture for image classification]


## 🧠 Model Architecture
The model uses a multi-layer CNN designed to extract hierarchical spatial features:
* **Convolutional Layers:** Four `Conv2D` layers with ReLU activation for feature detection.
* **Pooling:** `MaxPool2D` layers for spatial variance reduction.
* **Regularization:** `Dropout` layers (0.25 and 0.5) to improve model robustness and prevent overfitting.
* **Output:** A `Dense` layer with a **Softmax** activation function to output probabilities for 43 classes.

## 🛠️ Installation & Usage

1. Clone the Repository
```bash
git clone [https://github.com/Traffic-Sign-Recognition.git](https://github.com/Muqeet04/YTraffic-Sign-Recognition.git)
cd Traffic-Sign-Recognition

2. Install Dependencies
pip install streamlit tensorflow-cpu pillow numpy

3. Run the App
streamlit run app.py
