

# Landmark Detection using OpenCV, Dlib, and Mediapipe

## 1. Install Required Libraries
# pip install opencv-python dlib mediapipe numpy

import cv2
import dlib
import mediapipe as mp
import numpy as np

# Load face detector and landmark predictor (Dlib)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Function for Dlib landmark detection
def detect_landmarks_dlib(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(68):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image

# Function for Mediapipe landmark detection
def detect_landmarks_mediapipe(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                h, w, _ = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
    return image

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_dlib = detect_landmarks_dlib(frame.copy())
    frame_mediapipe = detect_landmarks_mediapipe(frame.copy())

    cv2.imshow("Dlib Landmarks", frame_dlib)
    cv2.imshow("Mediapipe Landmarks", frame_mediapipe)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

