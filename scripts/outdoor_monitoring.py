import cv2
import numpy as np
import os

# Load YOLO model
def load_yolo():
    # Paths to YOLO files
    yolo_dir = os.path.join(os.path.dirname(__file__), 'yolo')
    weights_path = os.path.join(yolo_dir, "yolov3.weights")
    cfg_path = os.path.join(yolo_dir, "yolov3.cfg")
    names_path = os.path.join(yolo_dir, "coco.names")

    # Check if files exist
    if not (os.path.isfile(weights_path) and os.path.isfile(cfg_path) and os.path.isfile(names_path)):
        raise FileNotFoundError("YOLO weights, config, or names file not found")

    # Load YOLO network
    net = cv2.dnn.readNet(weights_path, cfg_path)
    
    # Get YOLO layer names and output layer indices
    layer_names = net.getLayerNames()
    output_layer_indices = net.getUnconnectedOutLayers().flatten()
    output_layers = [layer_names[i - 1] for i in output_layer_indices]
    
    # Load class names
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, output_layers, classes

# Detect objects using YOLO
def detect_objects(frame, net, output_layers):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indexes, class_ids, confidences, boxes

# Main function
def main():
    net, output_layers, classes = load_yolo()
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        indexes, class_ids, confidences, boxes = detect_objects(frame, net, output_layers)

        if indexes is not None and len(indexes) > 0:
            indexes = indexes.flatten()
            for i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                confidence = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("No objects detected")

        cv2.imshow("Outdoor Monitoring", frame)

        # Check for ESC key with a slightly increased delay
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC key
            print("ESC key pressed. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
