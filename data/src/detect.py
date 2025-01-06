import cv2
import numpy as np

# Load YOLO
def load_model(model_path, config_path, labels_path):
    net = cv2.dnn.readNet(model_path, config_path)
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return net, labels

def detect_people(frame, net, labels, conf_threshold=0.5):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)
    
    results = []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold and labels[class_id] == 'person':
                box = detection[:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                results.append((x, y, int(box_width), int(box_height), confidence))
    return results
