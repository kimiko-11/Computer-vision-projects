import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pretrained model (or a custom one trained to detect bottles)
cap = cv2.VideoCapture("C:/Users/Kimaya/ComputerVision_projects/venv/code/bottles_on_belt.mp4")

unique_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame")
        break

    input_size = 640
    frame_resized = cv2.resize(frame, (input_size, input_size))

    # Run detection with tracking enabled
    results = model.track(frame_resized, classes=[39], persist=True, conf=0.25)  # track mode returns IDs

    annotated_frame = results[0].plot()

    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        for oid in ids:
            unique_ids.add(int(oid))

    cv2.putText(annotated_frame, f"Count: {len(unique_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Object Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
