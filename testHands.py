import cv2 as cv
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial.distance import euclidean
import math
import numpy as np
import os
import pickle as pick
from random import choice
from flask import Flask, render_template, Response

app = Flask(__name__)

GESTURE_DIR = "gestos_guardados"
os.makedirs(GESTURE_DIR, exist_ok=True)

# Inicializar MediaPipe
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Indices de landmarks para rostro
MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM = 61, 291, 13, 14
EYEBROWLEFT, EYELEFT, EYEBROWRIGHT, EYERIGHT = 52, 159, 33, 263
BARBILLA1, BARBILLA2, BARBILLA3 = 148, 152, 377
MANDIBULAIZQ, MANDIBULADER = 58, 288
NOSEBOTTOM, NOSETOP = 94, 6
EYELEFTBOTTOM = 145

# Distancia 2D
def distancia_2d(p1, p2):
    dx, dy = p2.x - p1.x, p2.y - p1.y
    return math.sqrt(dx**2 + dy**2)

# ---------------- FUNCIONES DE MANO ----------------
def is_fist(hand_landmarks):
    return all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y 
               for tip, dip in zip(
                   [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP],
                   [mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                    mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.PINKY_DIP]))

def is_peace(hand_landmarks):
    return (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y)

def is_thumb_up(hand_landmarks):
    return (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)

def is_rock(hand_landmarks):
    return (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y)

def is_extended(hand_landmarks):
    return all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y 
               for tip, dip in zip(
                   [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.THUMB_TIP],
                   [mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                    mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.THUMB_MCP]))

# ---------------- FLASK STREAM FUNCTION ----------------
def gen_frames():
    cam = cv.VideoCapture(0)
    while True:
        success, frame = cam.read()
        if not success:
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        hands_detected = hands.process(frame_rgb)
        face_detected = face.process(frame_rgb)
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

        # Detectar gestos de mano
        if hands_detected.multi_hand_landmarks:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
                if is_fist(hand_landmarks):
                    cv.putText(frame_bgr, "Puño cerrado", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif is_peace(hand_landmarks):
                    cv.putText(frame_bgr, "Paz y amor", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif is_thumb_up(hand_landmarks):
                    cv.putText(frame_bgr, "Pulgar arriba", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif is_rock(hand_landmarks):
                    cv.putText(frame_bgr, "Rock", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif is_extended(hand_landmarks):
                    cv.putText(frame_bgr, "Mano extendida", (10, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Detectar gestos de rostro
        if face_detected.multi_face_landmarks:
            landmarks = face_detected.multi_face_landmarks[0].landmark
            left, right, top, bottom = landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT], landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM]
            eyeleft, eyebrowleft, eyeright, eyebrowright = landmarks[EYELEFT], landmarks[EYEBROWLEFT], landmarks[EYERIGHT], landmarks[EYEBROWRIGHT]
            mandibulaIzq, mandibulaDer = landmarks[MANDIBULAIZQ], landmarks[MANDIBULADER]
            nosebottom, nosetop = landmarks[NOSEBOTTOM], landmarks[NOSETOP]
            eyeleftbottom = landmarks[EYELEFTBOTTOM]

            mouth_width, mouth_height = euclidean((left.x, left.y), (right.x, right.y)), euclidean((top.x, top.y), (bottom.x, bottom.y))
            ratio = mouth_width / mouth_height if mouth_height != 0 else 0
            dist_pupilas = distancia_2d(eyeright, eyeleft)
            dist_normalizada = distancia_2d(eyeleft, eyebrowleft) / dist_pupilas if dist_pupilas != 0 else 0
            dist_nose = distancia_2d(nosebottom, nosetop)
            distance_left_head = distancia_2d(left, mandibulaIzq)
            distance_right_head = distancia_2d(mandibulaDer, right)

            if ratio < 10:
                cv.putText(frame_bgr, "Sonrisa", (10, 180), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if dist_normalizada > 0.65:
                cv.putText(frame_bgr, "Ceja levantada", (10, 210), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if distance_left_head > dist_nose or distance_right_head > dist_nose:
                cv.putText(frame_bgr, "Cabeza de lado", (10, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if 1 < ratio < 4:
                cv.putText(frame_bgr, "Boca abierta", (10, 270), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv.imencode('.jpg', frame_bgr)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cam.release()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
