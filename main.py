import time
import cv2
import json
from ultralytics import YOLO


def detect_people(path: str, model_name: str, device: str):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30
    frame_step = int(fps * 10)
    results_data = {}
    current_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    failures = 0
    max_failures = 300
    total_inference_time = 0
    processed_frames = 0
    while cap.isOpened():
        if current_frame % frame_step == 0:
            success, frame = cap.read()
            if not success:
                failures += 1
                current_frame += 1
                if failures > max_failures:
                    break
                continue
            failures = 0
            current_time = int(current_frame / fps)
            start_time = time.time()
            results = model.predict(
                frame,
                classes=[0],
                conf=0.2,
                imgsz=640,
                device=device,
                verbose=False
            )
            end_time = time.time()
            total_inference_time += end_time - start_time
            processed_frames += 1
            people_count = len(results[0].boxes) if results[0].boxes else 0
            results_data[f"{current_time}s"] = people_count
        else:
            success = cap.grab()
            if not success:
                failures += 1
                current_frame += 1
                if failures > max_failures:
                    break
                continue
            else:
                failures = 0
        current_frame += 1
        if total_frames > 0 and current_frame >= total_frames:
            break

    cap.release()
    average_inference = total_inference_time / processed_frames if processed_frames > 0 else 0
    results_data["metrics"] = {
        "total_inference_time_sec": round(total_inference_time, 3),
        "average_time_per_frame_sec": round(average_inference, 4)
    }
    return results_data

    
def main():
    current_model = 'yolo26x.pt'
    current_device = 'cpu'
    stats = detect_people(path="data/output.mp4", model_name=current_model, device=current_device)
    clean_model_name = current_model.split('.')[0]
    filename = f'data/result_{clean_model_name}_{current_device}.json'
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(stats, file, indent=4)

if __name__ == "__main__":
    main()