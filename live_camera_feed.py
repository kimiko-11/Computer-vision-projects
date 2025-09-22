import cv2
from ultralytics import YOLO
# Load the YOLOv8 model
model=YOLO('yolov8n.pt')
cameraman=cv2.VideoCapture(0)
while True:
    ret,frame=cameraman.read()
    if not ret:
        print("failed to grab frame")
        break
    results=model(frame)
    annotated_img=results[0].plot()
    cv2.imshow('LIVE CAMERA FEED',annotated_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cameraman.release()
cv2.destroyAllWindows()
