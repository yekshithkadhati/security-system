import cv2
import numpy as np
from PIL import Image
import os

def train_model(data_dir):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for label_name in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        label_map[label_id] = label_name
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = Image.open(image_path).convert('L')
            image_np = np.array(image, 'uint8')
            faces_detected = face_cascade.detectMultiScale(image_np)
            for (x, y, w, h) in faces_detected:
                faces.append(image_np[y:y+h, x:x+w])
                labels.append(label_id)
        label_id += 1

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    
    # Ensure the 'models' directory exists under security_system
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the trained model and label map in the models directory
    face_recognizer.save(os.path.join(models_dir, 'face_recognizer.yml'))
    with open(os.path.join(models_dir, 'label_map.txt'), 'w') as f:
        for label_id, label_name in label_map.items():
            f.write(f"{label_id}:{label_name}\n")
        print("Trained sucessfully...")

if __name__ == "__main__":
    train_model('../data/train_set')  # Use 'train_set' for training images
