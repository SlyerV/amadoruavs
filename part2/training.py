import os
from ultralytics import YOLO

# Absolute Path
baseDir = os.path.abspath(os.path.dirname(__file__))
dataPath = os.path.join(baseDir, "data.yaml")

model = YOLO('yolov8n.pt')
model.train(data=dataPath, epochs=50, imgsz=320, batch=4, exist_ok=True) # Used low imgsz and batch for faster training on codespace - feel free to change parameters
