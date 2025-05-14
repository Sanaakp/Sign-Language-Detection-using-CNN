# Sign Language Detection using CNN

This project focuses on real-time American Sign Language (ASL) detection using a Convolutional Neural Network (CNN). It uses image data of various ASL hand signs for training and utilizes TensorFlow/Keras for model development and live prediction via webcam.

---

## Features
- Real-time ASL detection through webcam
- Trained on custom image dataset
- CNN architecture built with TensorFlow/Keras
- Saves trained model as `.h5` file
- User-friendly Python interface for testing

---

## Project Structure

├── code/
│ ├── model.py
│ ├── sign_detection.py
│ └── ...
├── dataset/
|   ├──test/
│     ├── A/
│     ├── B/
|     └── ...
|    ├──train/
│     ├── A/
│     ├── B/
|     └── ...
|    ├──val/
│     ├── A/
│     ├── B/
|     └── ...
├── asl_model.h5
└── README.md

## Requirements

- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy

** Future Work
.Add support for numbers and gestures

.Integrate with voice output

.Improve detection in low-light conditions
