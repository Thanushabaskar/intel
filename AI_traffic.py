import cv2
import torch
import numpy as np
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load video files
video_file1 = 'traffic1.mp4'
video_file2 = 'traffic2.mp4'

cap1 = cv2.VideoCapture(video_file1)
cap2 = cv2.VideoCapture(video_file2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Unable to open video files.")
    exit()

# Define counting line coordinates
counting_line_y = 300

# Initialize tracking dictionaries
tracked_vehicles = {
    'video1': {},
    'video2': {}
}

# Initialize vehicle counts
count1 = 0
count2 = 0

# Initialize traffic light states
light_state1 = "RED"
light_state2 = "GREEN"
pause_start_time = None  # Timer for the current pause
pause_duration = 10  # Duration for red and green light
yellow_duration = 5  # Duration for yellow light before green
paused_video1 = True  # Start with video1 paused
paused_video2 = False  # Start with video2 playing

# Store last processed frames
last_frame1 = None
last_frame2 = None

# Set both videos to the same frame size
frame_width = 800
frame_height = 600

# Start processing videos
while True:
    # Read frames from videos
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Process frames for video 1
    if not paused_video1 and ret1:
        frame1 = cv2.resize(frame1, (frame_width, frame_height))
        rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        results1 = model(rgb_frame1)

        # Extract bounding box coordinates and class IDs
        detections1 = results1.pandas().xyxy[0]

        # Draw bounding boxes and count vehicles for Video 1
        for index, row in detections1.iterrows():
            if row['name'] in ['car', 'truck', 'bus']:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                vehicle_id = f"car_{index}"
                if vehicle_id not in tracked_vehicles['video1']:
                    tracked_vehicles['video1'][vehicle_id] = {
                        'centroid': (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2),
                        'crossed': False
                    }
                # Check if the vehicle has crossed the counting line
                if not tracked_vehicles['video1'][vehicle_id]['crossed'] and y2 > counting_line_y:
                    tracked_vehicles['video1'][vehicle_id]['crossed'] = True
                    count1 += 1
                # Reset the counted status if the vehicle is no longer detected
                if y2 < counting_line_y:
                    tracked_vehicles['video1'][vehicle_id]['crossed'] = False

                cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)

        last_frame1 = frame1  # Store last processed frame

    # Use the last processed frame if video1 is paused
    elif paused_video1:
        if last_frame1 is not None:
            frame1 = last_frame1

    # Process frames for video 2
    if not paused_video2 and ret2:
        frame2 = cv2.resize(frame2, (frame_width, frame_height))
        rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        results2 = model(rgb_frame2)

        # Extract bounding box coordinates and class IDs
        detections2 = results2.pandas().xyxy[0]

        # Draw bounding boxes and count vehicles for Video 2
        for index, row in detections2.iterrows():
            if row['name'] in ['car', 'truck', 'bus']:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                vehicle_id = f"car_{index}"
                if vehicle_id not in tracked_vehicles['video2']:
                    tracked_vehicles['video2'][vehicle_id] = {
                        'centroid': (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2),
                        'crossed': False
                    }
                # Check if the vehicle has crossed the counting line
                if not tracked_vehicles['video2'][vehicle_id]['crossed'] and y2 > counting_line_y:
                    tracked_vehicles['video2'][vehicle_id]['crossed'] = True
                    count2 += 1
                # Reset the counted status if the vehicle is no longer detected
                if y2 < counting_line_y:
                    tracked_vehicles['video2'][vehicle_id]['crossed'] = False

                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)

        last_frame2 = frame2  # Store last processed frame

    # Use the last processed frame if video2 is paused
    elif paused_video2:
        if last_frame2 is not None:
            frame2 = last_frame2

    # Display vehicle counts on frames
    cv2.putText(frame1, f'Total Vehicles: {count1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame2, f'Total Vehicles: {count2}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Manage traffic signal alternation with yellow light
    if light_state1 == "RED":
        if pause_start_time is None:
            pause_start_time = time.time()  # Start countdown

        countdown = pause_duration - (time.time() - pause_start_time)
        if countdown <= 0:
            # Switch to yellow light for 5 seconds
            light_state1 = "YELLOW"
            pause_start_time = time.time()  # Reset timer for yellow light

        else:
            # Show countdown on frame1
            cv2.rectangle(frame1, (10, 90), (200, 130), (0, 0, 0), -1)  # Fill rectangle for clarity
            cv2.putText(frame1, f'Countdown: {int(countdown)}s', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif light_state1 == "YELLOW":
        countdown = yellow_duration - (time.time() - pause_start_time)
        if countdown <= 0:
            light_state1 = "GREEN"
            light_state2 = "RED"
            pause_start_time = None
            paused_video1 = False  # Resume video1
            paused_video2 = True  # Pause video2

        # Show yellow light without overlap
        cv2.putText(frame1, 'YELLOW LIGHT', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    elif light_state2 == "RED":
        if pause_start_time is None:
            pause_start_time = time.time()  # Start countdown

        countdown = pause_duration - (time.time() - pause_start_time)
        if countdown <= 0:
            light_state2 = "YELLOW"
            pause_start_time = time.time()  # Reset timer for yellow light

        else:
            # Show countdown on frame2
            cv2.rectangle(frame2, (10, 90), (200, 130), (0, 0, 0), -1)  # Fill rectangle for clarity
            cv2.putText(frame2, f'Countdown: {int(countdown)}s', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif light_state2 == "YELLOW":
        countdown = yellow_duration - (time.time() - pause_start_time)
        if countdown <= 0:
            light_state2 = "GREEN"
            light_state1 = "RED"
            pause_start_time = None
            paused_video1 = True  # Pause video1
            paused_video2 = False  # Resume video2

        # Show yellow light without overlap
        cv2.putText(frame2, 'YELLOW LIGHT', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the traffic light state
    cv2.putText(frame1, f'Traffic Light: {light_state1}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame2, f'Traffic Light: {light_state2}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frames
    cv2.imshow('Video 1', frame1)
    cv2.imshow('Video 2', frame2)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures and destroy windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
