import cv2
from ultralytics import YOLO
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
# Open the video file
cameraman= cv2.VideoCapture("C:/Users/Kimaya/ComputerVision_projects/venv/code/people_on_street.mp4")
while True:
    ret, frame = cameraman.read()
    if not cameraman.isOpened():
        print("Error: Could not open video.")
        exit()
    if not ret:
        print("Failed to grab frame or end of video reached")
        break
    input_size = 640  # common YOLOv8 input size
    frame_resized = cv2.resize(frame, (input_size, input_size))
    results = model(frame_resized, classes=[0], conf=0.25)
    annotated_img = results[0].plot()
    cv2.imshow('Multi-Object Detection Video', annotated_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cameraman.release()
cv2.destroyAllWindows()
