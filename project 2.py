import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import os
import pygame
import time 


pygame.mixer.init()

alert_sound = pygame.mixer.Sound("assets/beep.mp3")


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords



file_path = "assets/shape_predictor_68_face_landmarks.dat"


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(file_path)


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


EAR_THRESHOLD = 0.25           
EAR_CONSEC_FRAMES = 20         
ALERT_COOLDOWN_SEC = 2.0       

drowsy_counter = 0
last_alert_time = 0
alert_active = False
# -----------------------------------------------------------


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = shape_to_np(shape)

        left_eye = shape[42:48]
        right_eye = shape[36:42]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            drowsy_counter += 1

            # Show progress counter on screen
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Trigger only if it's low for enough frames
            if drowsy_counter >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "Drowsiness Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                current_time = time.time()
                if (current_time - last_alert_time) > ALERT_COOLDOWN_SEC:
                    alert_sound.play()
                    last_alert_time = current_time

                alert_active = True

        else:
            # Reset when eyes open again
            drowsy_counter = 0
            alert_active = False

            # show ear
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Driver Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
