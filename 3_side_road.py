import os  
import cv2  
import numpy as np  
from object_detection import ObjectDetection  

# Verify file paths  
cfg_path = "C:/proj/dnn_model/yolov4.cfg"  
weights_path = "C:/proj/dnn_model/yolov4.weights"  
classes_path = "C:/proj/dnn_model/classes.txt"  
video_path1 = "C:/proj/source_code/trafficvideo1.mp4"  
video_path2 = "C:/proj/source_code/trafficvideo2.mp4"  
video_path3 = "C:/proj/source_code/trafficvideo3.mp4"  # Add third video path

# Initialize Object Detection  
od = ObjectDetection(weights_path=weights_path, cfg_path=cfg_path, classes_path=classes_path)  

# Check if video files exist  
cap1 = cv2.VideoCapture(video_path1)  
cap2 = cv2.VideoCapture(video_path2)  
cap3 = cv2.VideoCapture(video_path3)  # Initialize third video capture

# Initialize variables for counting  
total_vehicle_count1 = 0  
total_vehicle_count2 = 0  
total_vehicle_count3 = 0  # Total count for third video
detected_centroids1 = []  # List to store unique centroids for video 1  
detected_centroids2 = []  # List to store unique centroids for video 2  
detected_centroids3 = []  # List to store unique centroids for video 3
vehicle_boxes1 = []  # Store bounding boxes for video 1  
vehicle_boxes2 = []  # Store bounding boxes for video 2  
vehicle_boxes3 = []  # Store bounding boxes for video 3
frame_height, frame_width = 480, 640  
skip_frames = 2  
delay = int(1000 / 30)  

# Initialize frame counters  
count1 = 0  
count2 = 0  
count3 = 0  # Frame counter for third video

def is_new_centroid(centroid, detected_centroids):  
    """  
    Check if the given centroid has been detected before.  
    """  
    for c in detected_centroids:  
        if abs(c[0] - centroid[0]) < 50 and abs(c[1] - centroid[1]) < 50:  
            return False  
    return True  

while True:  
    ret1, frame1 = cap1.read()  
    ret2, frame2 = cap2.read()  
    ret3, frame3 = cap3.read()  # Read frame from third video

    if not ret1 and not ret2 and not ret3:  
        break  

    # Process video 1  
    if ret1:  
        frame1 = cv2.resize(frame1, (frame_width, frame_height))  
        count1 += 1  

        if count1 % skip_frames == 0:  
            class_ids1, scores1, boxes1 = od.detect(frame1)  

            for box in boxes1:  
                (x, y, w, h) = box  
                centroid = (int(x + w // 2), int(y + h // 2))  # Calculate centroid  

                # Check if this centroid has been detected already  
                if is_new_centroid(centroid, detected_centroids1):  
                    detected_centroids1.append(centroid)  # Add centroid to the list  
                    total_vehicle_count1 += 1  # Increment total vehicle count  
                    vehicle_boxes1.append(box)  # Store the box for display  

            # Draw all bounding boxes for detected vehicles  
            for box in vehicle_boxes1:  
                (x, y, w, h) = box  
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green box  

        # Update and display the total vehicle count for video 1  
        cv2.putText(frame1, f"Vehicles: {total_vehicle_count1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  

    # Process video 2  
    if ret2:  
        frame2 = cv2.resize(frame2, (frame_width, frame_height))  
        count2 += 1  

        if count2 % skip_frames == 0:  
            class_ids2, scores2, boxes2 = od.detect(frame2)  

            for box in boxes2:  
                (x, y, w, h) = box  
                centroid = (int(x + w // 2), int(y + h // 2))  # Calculate centroid  

                # Check if this centroid has been detected already  
                if is_new_centroid(centroid, detected_centroids2):  
                    detected_centroids2.append(centroid)  # Add centroid to the list  
                    total_vehicle_count2 += 1  # Increment total vehicle count  
                    vehicle_boxes2.append(box)  # Store the box for display  

            # Draw all bounding boxes for detected vehicles  
            for box in vehicle_boxes2:  
                (x, y, w, h) = box  
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green box  

        # Update and display the total vehicle count for video 2  
        cv2.putText(frame2, f"Vehicles: {total_vehicle_count2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  

    # Process video 3  
    if ret3:  
        frame3 = cv2.resize(frame3, (frame_width, frame_height))  
        count3 += 1  

        if count3 % skip_frames == 0:  
            class_ids3, scores3, boxes3 = od.detect(frame3)  

            for box in boxes3:  
                (x, y, w, h) = box  
                centroid = (int(x + w // 2), int(y + h // 2))  # Calculate centroid  

                # Check if this centroid has been detected already  
                if is_new_centroid(centroid, detected_centroids3):  
                    detected_centroids3.append(centroid)  # Add centroid to the list  
                    total_vehicle_count3 += 1  # Increment total vehicle count  
                    vehicle_boxes3.append(box)  # Store the box for display  

            # Draw all bounding boxes for detected vehicles  
            for box in vehicle_boxes3:  
                (x, y, w, h) = box  
                cv2.rectangle(frame3, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green box  

        # Update and display the total vehicle count for video 3  
        cv2.putText(frame3, f"Vehicles: {total_vehicle_count3}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  

    # Compare counts and decide which video gets the green signal  
    max_count = max(total_vehicle_count1, total_vehicle_count2, total_vehicle_count3)  
    if max_count == total_vehicle_count1:  
        cv2.putText(frame1, "Traffic Light: GREEN", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
        cv2.putText(frame2, "Traffic Light: RED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        cv2.putText(frame3, "Traffic Light: RED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
    elif max_count == total_vehicle_count2:  
        cv2.putText(frame1, "Traffic Light: RED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        cv2.putText(frame2, "Traffic Light: GREEN", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
        cv2.putText(frame3, "Traffic Light: RED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
    elif max_count == total_vehicle_count3:  
        cv2.putText(frame1, "Traffic Light: RED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        cv2.putText(frame2, "Traffic Light: RED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        cv2.putText(frame3, "Traffic Light: GREEN", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  

    # Show frames  
    cv2.imshow("Video 1", frame1)  
    cv2.imshow("Video 2", frame2)  
    cv2.imshow("Video 3", frame3)  # Show third video

    # Break loop on 'q' key press  
    if cv2.waitKey(delay) & 0xFF == ord('q'):  
        break  

# Release video captures and close windows  
cap1.release()  
cap2.release()  
cap3.release()  # Release third video capture
cv2.destroyAllWindows()  
