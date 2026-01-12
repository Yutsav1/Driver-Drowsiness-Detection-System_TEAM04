from ultralytics import YOLO

# Load YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train model
model.train(
    data="yolo_config.yaml",   # Path to YAML file
    epochs=100,
    imgsz=640,
    batch=16
)
