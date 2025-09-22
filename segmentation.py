import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Load YOLO segmentation model
model = YOLO('yolov8n-seg.pt') 
cap = cv2.VideoCapture("C:/Users/Kimaya/ComputerVision_projects/people_in_park.mp4")

# Tracking data structures
appear = defaultdict(int)   # keep track of unique IDs
display_width = 800         # resize for display
next_id = 1                 # starting ID

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame")
        break

    # Resize frame
    h, w = frame.shape[:2]
    display_height = int((display_width / w) * h)
    frame_resized = cv2.resize(frame, (display_width, display_height))

    # YOLO segmentation + tracking
    results = model.track(frame_resized, classes=[0], persist=True, conf=0.25)  

    annotated_frame = frame_resized.copy()

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()     # (N, H, W)
        boxes = results[0].boxes.xyxy.cpu().numpy()     # (N, 4)

        for mask, box in zip(masks, boxes):
            # Convert mask back to image space
            mask_resized = cv2.resize(mask, (display_width, display_height))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            # Extract contours
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contour on frame
            cv2.drawContours(annotated_frame, contours, -1, (0, 255, 0), 2)

            # Center point of box
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Assign a new ID (demo, not real tracking yet)
            oid = next_id
            next_id += 1
            appear[oid] = 1

            cv2.putText(annotated_frame, f'ID:{oid}', (cx, cy - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show result
    cv2.imshow('Segmentation with Contours', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f'Total unique objects detected: {len(appear)}')
