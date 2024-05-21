import cv2
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture('./traffic.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    cv2.imshow('YOLO', results.render()[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
