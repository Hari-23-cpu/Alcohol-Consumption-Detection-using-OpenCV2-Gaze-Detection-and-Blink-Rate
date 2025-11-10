# Alcohol-Consumption-Detection-using-OpenCV2-Gaze-Detection-and-Blink-Rate
Developed a real-time system to detect alcohol influence using eye gaze, blink rate, and duration. Used OpenCV2 for facial feature extraction and gaze tracking to assess alertness

Overview

This project focuses on developing a real-time alcohol consumption detection system using OpenCV2. The system monitors a person’s eye gaze, blink rate, and blink duration to assess levels of alertness and potential alcohol influence.

By analyzing facial landmarks and eye behavior patterns, the model aims to provide a non-invasive and automated way to evaluate driver or user alertness in real-time environments.

Key Features:
Real-time face and eye detection using OpenCV2.
Gaze tracking to monitor focus and attention.
Blink rate and duration analysis to detect fatigue or possible alcohol influence.
Lightweight and easy-to-deploy model suitable for safety and surveillance systems.
Can be extended for driver monitoring or workplace safety applications.

Technologies Used:
Python
OpenCV2
Dlib / Mediapipe (optional) for facial landmark detection
NumPy
Matplotlib (for data visualization and analysis)

Working Principle:
Facial Feature Extraction:
Detects the eyes and key facial landmarks from a video feed or camera input.
Gaze Tracking:
Monitors eye direction to determine attention and focus deviation.
Blink Analysis:
Calculates blink frequency and duration; abnormal patterns may indicate alcohol influence.
Decision Logic:
Combines blink rate and gaze deviation metrics to assess alertness levels.
