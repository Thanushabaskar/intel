import cv2
import numpy as np

class ObjectDetection:
    def __init__(self, weights_path, cfg_path, classes_path):
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        self.classes = []
        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        scores = []
        boxes = []

        for out in outs:
            for detection in out:
                scores_data = detection[5:]
                class_id = np.argmax(scores_data)
                confidence = scores_data[class_id]
                if confidence > 0.5:  # Adjust the threshold as needed
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    scores.append(float(confidence))
                    class_ids.append(class_id)

        return class_ids, scores, boxes
