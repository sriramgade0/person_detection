import cv2

def start_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access camera.")
        return cap, False
    return cap, True

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        return None
    return frame

def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
