

# Object Detection using TensorFlow and OpenCV

## 1. Install Required Libraries
# pip install tensorflow opencv-python numpy

import tensorflow as tf
import cv2
import numpy as np

# Load a pre-trained TensorFlow object detection model
model = tf.saved_model.load("ssd_mobilenet_v2_fpnlite_320x320/saved_model")  # Download the model beforehand

# Load an image
image_path = "test_image.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = tf.convert_to_tensor(image_rgb)
input_tensor = input_tensor[tf.newaxis, ...]

# Perform Object Detection
detections = model(input_tensor)

# Process detections
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
boxes, classes, scores = detections["detection_boxes"], detections["detection_classes"], detections["detection_scores"]

# Draw bounding boxes on the image
height, width, _ = image.shape
for i in range(num_detections):
    if scores[i] > 0.5:  # Confidence threshold
        y1, x1, y2, x2 = boxes[i]
        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show the image
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

