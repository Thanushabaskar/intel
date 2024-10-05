AI-Based Smart Traffic Management System with OpenVINO Integration 

 Overview

This project implements an AI-Based Smart Traffic Management System using YOLO (You Only Look Once) for real-time vehicle detection and counting across three video feeds of a three-way intersection. The system dynamically adjusts traffic light signals based on vehicle counts to optimize traffic flow and reduce congestion. The project leverages OpenVINO to optimize the YOLO model for improved inference performance.

 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [OpenVINO Integration](#openvino-integration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

 Features

- Real-time vehicle detection and counting using YOLO
- Dynamic traffic light control based on vehicle counts from three video feeds
- Integration of OpenVINO for optimized inference performance
- Display of detected vehicle bounding boxes on the video feeds
- Easy-to-use interface with video playback and traffic signal status

 Technologies Used

- Python 3.8+
- OpenCV
- NumPy
- YOLO (v4)
- OpenVINO Toolkit
- Object detection libraries
- Video processing libraries

 Setup and Installation

 Prerequisites

- Python 3.8 or later
- OpenCV
- NumPy
- YOLO weights and configuration files
- OpenVINO Toolkit (latest version)
- Ensure video files are accessible

 Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-management-system.git
   cd traffic-management-system
   ```

2. Install required libraries:
   ```bash
   pip install opencv-python numpy
   ```

3. Download YOLO weights and configuration files. Place them in the `dnn_model` directory:
   - `yolov4.cfg`
   - `yolov4.weights`
   - `classes.txt`

4. Install OpenVINO:
   - Follow the [OpenVINO installation guide](https://docs.openvino.ai/latest/openvino_docs_installing_openvino_windows.html) for your operating system.
   - Ensure to set up the environment variables as instructed in the guide.

5. Convert YOLO model to OpenVINO format:
   - Use the Model Optimizer provided by OpenVINO to convert your YOLO model to an Intermediate Representation (IR) format:
     ```bash
     mo --input_model yolov4.weights --model_name yolov4 --framework darknet
     ```
   - This will generate `.xml` and `.bin` files in the specified directory.

6. Ensure video files are in the `source_code` directory:
   - `trafficvideo1.mp4`
   - `trafficvideo2.mp4`
   - `trafficvideo3.mp4`

OpenVINO Integration

Using OpenVINO for Inference

1. Modify the detection script to load the OpenVINO model instead of the original YOLO model. You can initialize the OpenVINO inference engine as follows:
   
   ```python
   from openvino.inference_engine import IECore

   # Load the model
   ie = IECore()
   net = ie.read_network(model='yolov4.xml', weights='yolov4.bin')
   exec_net = ie.load_network(network=net, device_name='CPU')
   ```

2. Run inference using the loaded OpenVINO model:
   
   ```python
   # Prepare the input blob
   input_blob = next(iter(net.input_info))
   output_blob = next(iter(net.outputs))
   frame = preprocess_frame(frame)  # Resize and preprocess your frame as required

   # Perform inference
   results = exec_net.infer(inputs={input_blob: frame})
   ```

3. Process results:  similarly as before to get vehicle detections, but now using the output from the OpenVINO model.

### Performance Improvements

Integrating OpenVINO optimizes the inference time, allowing for faster processing of video feeds. Ensure you test the performance gain compared to the original YOLO model.

 Usage

1. Run the script:
   ```bash
   python traffic_management.py
   ```

2. The application will open three video windows, displaying the traffic from each video feed, vehicle counts, and traffic light status.

3. Press `q` to exit the application.

Project Structure

```
traffic-management-system/
│
├── dnn_model/
│   ├── yolov4.cfg
│   ├── yolov4.weights
│   ├── yolov4.xml
│   └── yolov4.bin
│
├── source_code/
│   ├── trafficvideo1.mp4
│   ├── trafficvideo2.mp4
│   └── trafficvideo3.mp4
│
├── traffic_management.py
└── README.md
```
Methodology
Data Acquisition:

Video feeds from three different angles of the intersection are captured. These video files are processed to analyze traffic conditions continuously.
Object Detection:

The YOLO model is employed to detect vehicles in each frame of the video. The detection process involves:
Loading the YOLO model and configuration files.
Processing each video frame, resizing it for consistent input dimensions.
Running the model to identify vehicles and their bounding boxes.
Centroid Tracking:

For each detected vehicle, the centroid (geometric center) is calculated. A list of unique centroids is maintained to avoid double counting vehicles.
Vehicle Counting:

As new centroids are detected that haven't been recorded before, they are added to the count, allowing the system to track the total number of vehicles for each video feed.
Traffic Signal Management:

Based on the vehicle counts from the three video feeds, the system dynamically decides which direction should have a green signal. This decision-making process involves comparing the vehicle counts and displaying corresponding traffic light statuses on the video feeds:
If one direction has significantly more vehicles than the others, that direction receives a green light, while the other directions receive a red light.
In case of equal counts, all directions are given a red light, maintaining safety.
Performance Optimization:

The integration of OpenVINO enhances the performance of the YOLO model, reducing inference time and allowing for smoother real-time processing. The model is converted into an Intermediate Representation (IR) format suitable for OpenVINO, and inference is performed using the optimized engine.
Expected Outcomes
Enhanced Traffic Management: The system is expected to improve traffic flow and reduce waiting times at intersections, leading to more efficient road usage.
Real-Time Adaptability: By continuously analyzing traffic conditions and adjusting signals accordingly, the system can respond dynamically to changing traffic patterns.
Scalability: The architecture can be extended to include more cameras or integrate additional functionalities, such as pedestrian detection or emergency vehicle prioritization.
Demonstration of AI Capabilities: This project serves as a demonstration of how AI and machine learning can be applied in real-world scenarios to solve complex urban challenges, such as traffic congestion.

