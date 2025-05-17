import cv2
import numpy as np
import os
import time
from pushbullet import Pushbullet

# Initialize Pushbullet with your access token
pb = Pushbullet("o.jxvXLh3UaUvUyoyQU4sdgbexXG92JUi2")

# Load trained face recognition model
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(os.path.join(models_dir, 'face_recognizer.yml'))

# Load label map
label_map = {}
with open(os.path.join(models_dir, 'label_map.txt'), 'r') as f:
    for line in f:
        label_id, label_name = line.strip().split(':')
        label_map[int(label_id)] = label_name

# Initialize camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_notified = {}
notification_interval = 60  # seconds

def send_notification(person_name):
    print(f"Sending notification for {person_name}")
    pb.push_note("Security Alert", f"{person_name} has been detected.")

# Perform face recognition
while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face_roi)

        # Print debug information
        print(f"Face detected with label: {label}, confidence: {confidence}")

        # Adjust confidence threshold if necessary
        if confidence < 100:  # You might need to experiment with this threshold
            person_name = label_map.get(label, "Unknown")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            print(f"Detected {person_name} with confidence {confidence}")

            # Send notification if not already sent recently
            current_time = time.time()
            if person_name not in last_notified or (current_time - last_notified[person_name]) > notification_interval:
                send_notification(person_name)
                last_notified[person_name] = current_time
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print(f"Detected an unknown person with confidence {confidence}")

    cv2.imshow('Indoor Camera', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

camera.release()
cv2.destroyAllWindows()
