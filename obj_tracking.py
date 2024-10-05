import os
import cv2
import numpy as np
from object_detection import ObjectDetection

# Verify file paths
cfg_path = "C:/project/source_code/dnn_model/yolov4.cfg"
weights_path = "C:/project/source_code/dnn_model/yolov4.weights"
classes_path = "C:/project/source_code/dnn_model/classes.txt"
video_path = "C:/project/source_code/los_angeles.mp4"  # Updated video path

print("Checking file paths:")
print("Config file exists:", os.path.isfile(cfg_path))
print("Weights file exists:", os.path.isfile(weights_path))
print("Classes file exists:", os.path.isfile(classes_path))

# Initialize Object Detection
if os.path.isfile(cfg_path) and os.path.isfile(weights_path) and os.path.isfile(classes_path):
    od = ObjectDetection(weights_path=weights_path, cfg_path=cfg_path, classes_path=classes_path)
else:
    print("Error: One or more required files are missing.")
    exit()

# Check if video file exists
if not os.path.isfile(video_path):
    print(f"Error: Could not open video file at {video_path}.")
    exit()

cap = cv2.VideoCapture(video_path)

# Initialize variables
count = 0
center_points = []

while True:
    # Read frame
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Detect objects on frame
    class_ids, scores, boxes = od.detect(frame)

    # Draw bounding boxes and count vehicles
    vehicle_count = 0
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points.append((cx, cy))
        
        # Assuming vehicle classes have IDs in a specific range (e.g., 2 for car, 5 for bus, etc.)
        vehicle_count += 1  # Increment count for each detected vehicle
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the vehicle count on the top left corner
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for exit key press
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
