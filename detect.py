import cv2
import torch

# Load the YOLOv5 model with the correct path
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/User/Downloads/yolov5l-seg.pt', force_reload=True)

# Initialize video capture
cap = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Extract detections
    detections = results.pandas().xyxy[0]  # Get detections as a DataFrame

    # Draw bounding boxes on the frame
    for _, row in detections.iterrows():
        class_id = int(row['class'])
        confidence = row['confidence']
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = f"{results.names[class_id]} {confidence:.2f}"

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv5 Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
