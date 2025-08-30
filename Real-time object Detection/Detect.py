import cv2
from ultralytics import YOLO

# Load pre-trained YOLO model
model = YOLO("yolov8n.pt")   # 'n' = nano model, faster for real-time

# Open webcamrrrg
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    
    # Show results
    annotated_frame = results[0].plot()
    cv2.imshow("Real-Time Object Detection", annotated_frame)
    
    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
