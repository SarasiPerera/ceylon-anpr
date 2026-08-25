from ultralytics import YOLO
import cv2
import os

def detect_plate(image_path, output_dir="outputs"):
    # Load pretrained YOLOv8 model (general object detection for now)
    model = YOLO("yolov8n.pt")  # downloads automatically on first run

    results = model(image_path)

    os.makedirs(output_dir, exist_ok=True)

    for result in results:
        annotated = result.plot()
        filename = os.path.basename(image_path)
        out_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}")

    return results

if __name__ == "__main__":
    detect_plate("sample_images/test1.jpg")