import cv2
import json
from collections import defaultdict
from ultralytics import YOLO

def unique_people(path: str):
    model = YOLO('yolo26x.pt')
    cap = cv2.VideoCapture(path)
    people = set()
    track_history = defaultdict(int)
    skip_frame = 5
    current_frame = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        current_frame += 1
        if current_frame % skip_frame == 0:
            results = model.track(
                frame,
                classes=[0],
                persist=True,
                conf=0.2,
                imgsz=640,
                tracker='data/updated_bytetrack.yaml',
                verbose=False
            )
            if results[0].boxes is not None and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                for track_id in track_ids:
                    track_history[track_id] += 1
                    if track_history[track_id] > 5:
                        people.add(track_id)
    cap.release()
    return people

def main():
    people = unique_people(path="data/output.mp4")
    result_data = {"amount": len(people)}
    with open("data/result.json", "w", encoding="utf-8") as file:
        json.dump(result_data, file, indent=4)

if __name__ == "__main__":
    main()


