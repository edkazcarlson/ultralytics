from ultralytics import YOLO

# Load YOLOv10n model from scratch
model = YOLO("yolov10x.pt")

# Train the model
model.train(data = '/home/ecarlson/Downloads/multi-instance-object-detection-challenge/Starter_Dataset/yolo_params.yaml', epochs=100, batch=8)