import cv2
import numpy as np
import os
from sklearn.metrics import accuracy_score

def load_model(models_dir):
    # Load trained face recognition model
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(os.path.join(models_dir, 'face_recognizer.yml'))

    # Load label map
    label_map = {}
    with open(os.path.join(models_dir, 'label_map.txt'), 'r') as f:
        for line in f:
            label_id, label_name = line.strip().split(':')
            label_map[int(label_id)] = label_name
    
    return face_recognizer, label_map

def evaluate_model(test_dir, face_recognizer, label_map):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    true_labels = []
    predicted_labels = []

    for label_name in os.listdir(test_dir):
        label_path = os.path.join(test_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        label_id = [id for id, name in label_map.items() if name == label_name][0]

        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                label, _ = face_recognizer.predict(face_roi)
                true_labels.append(label_id)
                predicted_labels.append(label)

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    models_dir = '../models'
    test_dir = '../data/test_set'  # Update this path to your test dataset directory
    
    face_recognizer, label_map = load_model(models_dir)
    evaluate_model(test_dir, face_recognizer, label_map)
