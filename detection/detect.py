from ultralytics import YOLO


def load_yolo_model(model_path):
    # Load the YOLO model
    model = YOLO(model_path)  # Load the pretrained YOLOv8n model
    return model


def run_yolo_inference(model, image_path, img_size=640, confidence=0.6, iou_threshold=0.5, save=False):
    # Run inference on the specified image
    results = model.predict(image_path, imgsz=img_size, conf=confidence, iou=iou_threshold, save=save)

    return results
