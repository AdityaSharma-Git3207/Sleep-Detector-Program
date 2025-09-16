import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import os
import pygame
from time import time

# Initialize pygame mixer
pygame.mixer.init()

# Load alert sound
alert_sound = pygame.mixer.Sound("C:/Users/Utkarsh/Downloads/audio.wav")

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

# Get the absolute path to the file
file_path = os.path.abspath("C:/Users/Utkarsh/Downloads/shape_predictor_68_face_landmarks.dat")

# Load the face detector and facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(file_path)
 
# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Calculate the Euclidean distances between the vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Calculate the Euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])

    # Calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Load the video capture
cap = cv2.VideoCapture(0)

# Initialize variables
start_time = None
eyes_closed = False

while True:
    # Read a frame from the video capture
    ret, frame = cap.read() 

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        shape = predictor(gray, face)
        shape = shape_to_np(shape)

        # Extract the left and right eye landmarks
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate the EAR for each eye
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EAR for both eyes
        avg_ear = (left_ear + right_ear) / 2.0

        # Set a threshold for detecting drowsiness
        threshold = 0.25

        # Check if the EAR is below the threshold
        if avg_ear < threshold:
            if start_time is None:
                start_time = time()
            elif time() - start_time > 1:
                if not eyes_closed:
                    # Play alert sound
                    alert_sound.play()
                    eyes_closed = True
                cv2.putText(frame, "Drowsiness Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            start_time = None
            eyes_closed = False

    # Display the frame
    cv2.imshow("Driver Fatigue Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
