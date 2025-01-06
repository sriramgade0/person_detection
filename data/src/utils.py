import cv2

def draw_bounding_box(frame, x, y, w, h, confidence, label="Person", color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box with a label and confidence score on a frame.

    Args:
        frame (ndarray): The video frame on which to draw.
        x, y (int): Top-left corner of the bounding box.
        w, h (int): Width and height of the bounding box.
        confidence (float): Confidence score of the detection.
        label (str): Label of the detection (default is "Person").
        color (tuple): Color of the bounding box in BGR (default is green).
        thickness (int): Thickness of the bounding box border (default is 2).
    """
    # Draw the rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Prepare the label text
    text = f"{label}: {confidence:.2f}"
    
    # Calculate text size and background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        frame,
        (x, y - text_height - 5),
        (x + text_width, y + baseline - 5),
        color,
        -1,  # Filled rectangle for label background
    )

    # Put the text
    cv2.putText(
        frame,
        text,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),  # White text
        1,
    )

def preprocess_frame(frame, target_size=(416, 416)):
    """
    Preprocesses a frame for the detection model.

    Args:
        frame (ndarray): Input video frame.
        target_size (tuple): Target size for resizing (default is 416x416 for YOLO).

    Returns:
        blob (ndarray): Preprocessed frame blob.
    """
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, target_size, swapRB=True, crop=False)
    return blob

def non_max_suppression(boxes, confidences, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Applies Non-Maximum Suppression to filter overlapping bounding boxes.

    Args:
        boxes (list): List of bounding boxes [x, y, w, h].
        confidences (list): List of confidence scores for the bounding boxes.
        confidence_threshold (float): Minimum confidence score to keep a box.
        nms_threshold (float): Threshold for NMS IoU.

    Returns:
        indices (list): Indices of the filtered bounding boxes.
    """
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    return indices
