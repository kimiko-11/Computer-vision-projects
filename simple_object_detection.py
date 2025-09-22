import cv2
from ultralytics import YOLO
#Load the YOLO model
model=YOLO('yolov8n.pt')
image=cv2.imread('woman1.jpg')
results=model(image)
annotated_img=results[0].plot()
cv2.imshow('Annotated Image',annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()