from ultralytics import YOLO

# Load YOLOv8n model and export to ONNX
model = YOLO('yolov8n.pt')
model.export(format="onnx")
