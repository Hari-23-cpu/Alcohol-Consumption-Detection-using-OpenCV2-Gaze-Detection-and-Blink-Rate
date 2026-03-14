
## 📌 Alcohol Consumption Detection System

### 👁️ Real-time Monitoring via OpenCV2, Gaze Tracking, and Blink Analysis

This project develops a **real-time system to detect potential alcohol influence** by monitoring **physiological ocular signals**.
By analyzing **facial landmarks and eye behavior patterns**, the model provides a **non-invasive and automated method** to evaluate **driver or user alertness** in real-time environments.

---

### 🖼 System Interface

![Alcohol Detection Project](https://raw.githubusercontent.com/Hari-23-cpu/Alcohol-Consumption-Detection-using-OpenCV2-Gaze-Detection-and-Blink-Rate/main/Screenshot%202025-12-24%20201340.png)

**Figure 1:** System interface showing **real-time blink rate, gaze stability, and alertness status**.

---

# 🚀 Key Features

### ⚡ Real-time Detection

High-speed **face and eye tracking** using
🔗 [https://opencv.org/](https://opencv.org/)

### 👁️ Gaze Tracking

Monitors **pupil position** to evaluate **focus and attention deviation**.

### 👀 Blink Biometrics

Analyzes both:

* **Blink Rate**
* **Long Blink Duration**

These indicators help detect **fatigue or alcohol-induced impairment**.

### 🪶 Lightweight Architecture

Optimized for **real-time deployment** in safety systems without heavy GPU usage.

### 📈 Scalability

The system can be extended for:

* 🚗 **Driver Monitoring Systems (DMS)**
* 🏭 **Workplace Safety Monitoring**
* 🧠 **Fatigue Detection Systems**

---

# 🛠 Technologies Used

### 🐍 Python

Core programming language for **system logic and integration**
🔗 [https://www.python.org/](https://www.python.org/)

### 👁️ OpenCV

Computer vision and image processing
🔗 [https://opencv.org/](https://opencv.org/)

### 🧠 Dlib / MediaPipe

Facial landmark detection (**68-point face landmarks**)
🔗 [https://mediapipe.dev/](https://mediapipe.dev/)

### 🔢 NumPy

Numerical computation for similarity scores and gaze stability
🔗 [https://numpy.org/](https://numpy.org/)

### 📊 Matplotlib

Visualization for **post-session data analysis**
🔗 [https://matplotlib.org/](https://matplotlib.org/)

---

# 🧠 Working Principle

The system follows a **multi-stage pipeline** to convert raw webcam frames into an **Alertness Assessment**.

---

## 1️⃣ Facial Feature Extraction

The system detects:

* 👁️ Eyes
* 🙂 Facial landmarks

Using these landmarks, the **Eye Aspect Ratio (EAR)** is calculated to determine if the eyes are **open or closed**.

---

## 2️⃣ Gaze Tracking & Stability

The system evaluates **eye direction and focus**.

Example output:

```
Gaze Stability Score: 0.76
```

Lower values may indicate:

* Wandering focus
* Reduced muscle control
* Potential impairment

---

## 3️⃣ Blink Analysis

The system monitors two key blink metrics:

### 🔹 Blink Rate

Example:

```
5.73 blinks/min
```

### 🔹 Blink Duration

Detects **Long Blinks**, which can indicate:

* Alcohol influence
* Drowsiness
* Reduced alertness

---

## 4️⃣ Decision Logic

The final decision combines multiple signals:

* **Similarity Score**
* **Blink Rate**
* **Gaze Deviation**
* **Long Blink Detection**

If thresholds are exceeded, the system generates alerts such as:

```
⚠ Long Blink Detected
⚠ Alcohol: High
```
# 📂 Project Repository

🔗 [https://github.com/Hari-23-cpu/Alcohol-Consumption-Detection-using-OpenCV2-Gaze-Detection-and-Blink-Rate](https://github.com/Hari-23-cpu/Alcohol-Consumption-Detection-using-OpenCV2-Gaze-Detection-and-Blink-Rate)

