import cv2
from ultralytics import YOLO

# Load YOLOv8 small model (downloads automatically if not present)
model = YOLO("yolov8n.pt")

def detect_objects(frame):
    """
    Detect objects in a frame using YOLOv8.
    Returns: (annotated_frame, objects)
    """
    results = model(frame)  # Run YOLO inference
    annotated_frame = results[0].plot()  # Draw boxes on frame

    objects = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])     # Class ID
        label = model.names[cls_id]  # Class name (e.g., person, car)
        conf = float(box.conf[0])    # Confidence score
        objects.append({"label": label, "confidence": conf})

    return annotated_frame, objects
