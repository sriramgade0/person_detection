import cv2
from data.src.camera import start_camera, capture_frame, release_camera
from data.src.detect import load_model, detect_people

# Paths to models and labels
MODEL_PATH = "data/models/yolov3.weights"
CONFIG_PATH = "data/models/yolov3.cfg"
LABELS_PATH = "data/models/coco.names"

def main():
    net, labels = load_model(MODEL_PATH, CONFIG_PATH, LABELS_PATH)
    cap, success = start_camera()
    if not success:
        return
    
    while True:
        frame = capture_frame(cap)
        if frame is None:
            break
        
        detections = detect_people(frame, net, labels)
        
        # Draw detections
        for (x, y, w, h, conf) in detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"Person: {conf:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    release_camera(cap)

if __name__ == "__main__":
    main()
