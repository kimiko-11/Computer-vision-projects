from collections import defaultdict, deque
import cv2
from ultralytics import YOLO
import numpy as np

# ----------------------
# Setup
# ----------------------
model = YOLO('yolov8n.pt')  # YOLOv8 small model
cap = cv2.VideoCapture("C:\\Users\\Kimaya\\ComputerVision_projects\\venv\\code\\Cylists.mp4")  # Your video file

# Tracking data structures
trail = defaultdict(lambda: deque(maxlen=30))  # Stores past positions for trails
appear = defaultdict(int)                       # Keeps track of unique IDs

# Resize settings
display_width = 800  # width to resize frames for display

# ----------------------
# Main loop
# ----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame")
        break

    # Resize frame for display (maintain aspect ratio)
    height, width = frame.shape[:2]
    display_height = int((display_width / width) * height)
    frame_resized = cv2.resize(frame, (display_width, display_height))

    result = model.track(frame_resized, classes=[1], persist=True, conf=0.05)
    annotated_frame = result[0].plot()

    # Process detected boxes
    if result[0].boxes and result[0].boxes.id is not None:
        boxes = result[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        ids = result[0].boxes.id.cpu().numpy()      # YOLO object IDs

        for box, oid in zip(boxes, ids):
            oid = int(oid)
            # Center point of the box for trail
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            trail[oid].append((cx, cy))
            appear[oid] = 1

            # Draw trail
            for i in range(1, len(trail[oid])):
                cv2.line(annotated_frame, trail[oid][i-1], trail[oid][i], (0,255,0), 2)

            # Annotate ID
            cv2.putText(annotated_frame, f'ID:{oid}', (int(box[0]), int(box[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    # Display annotated frame
    cv2.imshow('Bottles with Trail', annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------
# Cleanup
# ----------------------
cap.release()
cv2.destroyAllWindows()
print(f'Total unique bottles detected: {len(appear)}')

