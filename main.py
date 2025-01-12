import cv2
from ultralytics import YOLO

# Load YOLOv8 model
MODEL_PATH = "yolov8n.pt"  # Use pre-trained YOLOv8 weights
model = YOLO(MODEL_PATH)

def main():
    cap = cv2.VideoCapture(0)  # Start the camera (use index 0 for default camera)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Use YOLOv8 for person detection
        results = model(frame)

        # Parse and draw detections on the frame
        for result in results[0].boxes:
            xyxy = result.xyxy[0].numpy().astype(int)  # Bounding box coordinates
            conf = result.conf[0].item()  # Confidence
            label = result.cls[0]  # Class index

            if int(label) == 0:  # Class 0 corresponds to 'person' in COCO
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"Person: {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Camera", frame)
        cv2.imwrite("frame.jpg", frame)
        print("Frame saved!")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
