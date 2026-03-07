Alcohol Consumption Detection SystemReal-time Monitoring via OpenCV2, Gaze Tracking, and Blink AnalysisThis project develops a real-time system to detect potential alcohol influence by monitoring physiological ocular signals. By analyzing facial landmarks and eye behavior patterns, the model provides a non-invasive, automated way to evaluate driver or user alertness in real-time environments.
Figure 1: System interface showing real-time blink rate, gaze stability, and alertness status.🚀 Key FeaturesReal-time Detection: High-speed face and eye tracking using OpenCV2.
Gaze Tracking: Monitors pupil position to evaluate focus and attention deviation.Blink Biometrics: Analyzes both frequency (Blink Rate) and duration (Long Blinks) to detect fatigue or alcohol-induced impairment.
Lightweight Architecture: Optimized for deployment in safety and surveillance systems without requiring heavy GPU resources.
Scalability: Can be extended for driver monitoring (DMS) or workplace safety checkpoints.
🛠 Technologies UsedPython: Core logic and system integration.
OpenCV2: Image processing and computer vision.
Dlib / Mediapipe: Facial landmark detection (identifying 68-point landmarks).
NumPy: Mathematical analysis of similarity scores and gaze stability.
Matplotlib: Data visualization for post-session analysis.
🧠 Working PrincipleThe system operates through a specialized pipeline to convert raw video frames into an "Alertness Assessment":
1. Facial Feature ExtractionThe system detects the eyes and key facial landmarks. By isolating the ocular region, it can calculate the Eye Aspect Ratio (EAR) to determine if the eyes are open or closed.
2. Gaze Tracking & StabilityThe model monitors eye direction to determine attention levels. It calculates a Gaze Stability Score (as seen in the shell output: 0.76); a lower score typically indicates wandering focus or lack of muscle control.3. Blink AnalysisBlink Rate: Calculates frequency (e.g., 5.73 blinks/min).
3.Duration: Detects "Long Blinks." Alcohol often slows the central nervous system, leading to longer eye-closure durations compared to a sober baseline.
4. Decision LogicThe system combines the Similarity Score, Blink Rate, and Gaze Deviation metrics. If the metrics cross a specific threshold, the system flags the user (e.g., "Long blink detected" or "Alcohol: High").
![Alcohol Detection Project]((https://github.com/Hari-23-cpu/Alcohol-Consumption-Detection-using-OpenCV2-Gaze-Detection-and-Blink-Rate/edit/main/))
