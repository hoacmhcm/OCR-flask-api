from ultralytics import YOLO

def run_yolo_inference(model_path, image_path, img_size=640, confidence=0.6, iou_threshold=0.5):
    # Load the YOLO model
    model = YOLO(model_path)  # Load the pretrained YOLOv8n model

    # Run inference on the specified image
    results = model.predict(image_path, save=True, imgsz=img_size, conf=confidence, iou=iou_threshold)

    return results
