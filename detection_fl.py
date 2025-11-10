import cv2
import numpy as np
import time
from gaze_tracking import GazeTracking

gaze = GazeTracking()
print("GazeTracking imported successfully!")

template = cv2.imread("C:/Users/krish/Downloads/eye.jpeg", cv2.IMREAD_GRAYSCALE)
template = cv2.resize(template, (24, 24))
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

blink_count = 0
eye_state_prev = 1
blink_start_frame = None
frame_counter = 0
start_time = time.time()
SIM_THRESOLD = 0.2
ALERT_BLINK_RATE_LOW = 5
ALERT_DURATION = 0.30
long_blink = False

gaze_blinking = 0
gaze_left = 0
gaze_right = 0
gaze_center = 0
gaze_unknown = 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 5)
    print(f"Eyes detected: {len(eyes)}")

    for (x, y, w, h) in eyes:
        eye_roi = gray[y:y + h, x:x + w]
        eye_resized = cv2.resize(eye_roi, (24, 24))
        res = cv2.matchTemplate(eye_resized, template, cv2.TM_CCOEFF_NORMED)
        similarity = res[0][0]
        print(f"Similarity score: {similarity:.2f}")
        eye_state = 1 if similarity > SIM_THRESOLD else 0

        if eye_state == 0:
            if blink_start_frame is None:
                blink_start_frame = time.time()
                frame_counter = 1
            else:
                frame_counter += 1
        else:  # Eye open
            if frame_counter >= 2:
                 blink_duration = time.time() - blink_start_frame
                 if blink_duration >= ALERT_DURATION:
                     blink_count+=1
                     long_blink=True
            blink_start_frame = None
            frame_counter = 0

        eye_state_prev = eye_state
        color = (0, 255, 0) if eye_state == 1 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Gaze tracking
    gaze.refresh(frame)
    gaze_frame = gaze.annotated_frame()

    if gaze.is_blinking():
        gaze_blinking += 1
        gaze_text = "eye blinking"
    elif gaze.is_left():
        gaze_left += 1
        gaze_text = "eye turning left"
    elif gaze.is_right():
        gaze_right += 1
        gaze_text = "eye turning right"
    elif gaze.is_center():
        gaze_center += 1
        gaze_text = "eye in straight"
    else:
        gaze_unknown += 1
        gaze_text = "Unknown"

    total_gaze = gaze_blinking + gaze_left + gaze_right + gaze_center + gaze_unknown
    gaze_stability = gaze_center / total_gaze if total_gaze > 0 else 0

    elapsed_time = time.time() - start_time
    blink_rate = blink_count / (elapsed_time / 60)

    print(f"Elapsed: {elapsed_time:.2f}, Blinks: {blink_count}, Blink rate: {blink_rate:.2f}, Total Gaze: {total_gaze}, Gaze stability: {gaze_stability:.2f}")

    if elapsed_time >= 15 and blink_count >= 1 and total_gaze > 20:
        if blink_rate < ALERT_BLINK_RATE_LOW or gaze_stability < 0.40:
            alcohol_alert = "Consumed Alcohol"
        elif blink_rate < ALERT_BLINK_RATE_LOW or gaze_stability < 0.40 :
            alcohol_alert = "Possibly drunken"
        else:
            alcohol_alert = "Normal"
    else:
        alcohol_alert = "Analyzing..."

    
    cv2.putText(gaze_frame, f"Blinks: {blink_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(gaze_frame, f"Blink rate: {blink_rate:.2f}/min", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if long_blink:
        cv2.putText(gaze_frame, "Long blink detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(gaze_frame, f"Gaze: {gaze_text}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(gaze_frame, f"Alcohol: {alcohol_alert}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Alcohol consumption", gaze_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
