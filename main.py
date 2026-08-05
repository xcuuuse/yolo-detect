import cv2

from ultralytics import YOLO

model = YOLO('yolo26m.pt')
path = "output.mp4"

cap = cv2.VideoCapture(path)
cv2.namedWindow("YOLO Inference", cv2.WINDOW_NORMAL)

cv2.resizeWindow("YOLO Inference", 1280, 720)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    results = model.track(frame, classes=[0], persist=True, conf=0.5, tracker='bytetrack.yaml')
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
