# facial-express-detection-using-facial-land-marks
import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame


pygame.mixer.init()

sleep_sound = pygame.mixer.Sound("/Users/bharathreddy/Downloads/sleep detection/sleep.wav")
drowsy_sound = pygame.mixer.Sound("/Users/bharathreddy/Downloads/sleep detection/error-2-126514.wav")


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/bharathreddy/Downloads/sleep detection/shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

face_frame = None

try:
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            face_frame = frame.copy()
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 3:
                    status = "SLEEPING !!!"
                    color = (255, 0, 0)
                    pygame.mixer.Sound.play(sleep_sound)
            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 3:
                    status = "Drowsy !"
                    color = (0, 0, 255)
                    pygame.mixer.Sound.play(drowsy_sound)
            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 3:
                    status = "Active :)"
                    color = (0, 255, 0)

            cv2.putText(face_frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            for (x, y) in landmarks:
                cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("Frame", frame)
        if face_frame is not None:
            cv2.imshow("Result of detector", face_frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")

cap.release()
cv2.destroyAllWindows()
