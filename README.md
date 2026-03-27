# Sign Language Transformer
A Real-time Sign Language Recognition System built using Deep Learning (LSTM) and MediaPipe Holistic.

## Overview
This project detects and classifies sign language gestures in real-time through a webcam. It uses **MediaPipe** to extract keypoints from hands, face, and pose, and an **LSTM (Long Short-Term Memory)** neural network to predict sequences of gestures.

##  Tech Stack
* **Python** - Core Programming
* **TensorFlow/Keras** - Deep Learning Model (LSTM)
* **MediaPipe** - Real-time Landmark Detection
* **OpenCV** - Image Processing & Webcam Feed
* **NumPy** - Data Manipulation

##  Project Structure
* `collect_data.py`: Script to capture and save landmark data for training.
* `train_model.py`: Defines and trains the LSTM model on captured sequences.
* `inference.py`: Real-time testing script using the webcam.
* `action.h5`: The pre-trained weights for the trained model.

##  Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AliHamza-codes/Sign-Language-Transformer.git](https://github.com/AliHamza-codes/Sign-Language-Transformer.git)
