import cv2
import mediapipe as mp
import base64
import numpy as np
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Inicialización de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Función para inicializar el procesamiento de la cámara
def initialize_camera():
    return cv2.VideoCapture(0)

# Función para procesar el frame y detectar manos y rostro
def process_frame(frame, hands, face_mesh):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hands_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)

    # Dibuja las manos detectadas
    if hands_results.multi_hand_landmarks:
        for landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Dibuja los puntos faciales
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    return frame

# Función para convertir el frame en base64
def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    return base64.b64encode(frame_bytes).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/capture_gestures")
def capture_gestures():
    cap = initialize_camera()
    hands = mp_hands.Hands()
    face_mesh = mp_face_mesh.FaceMesh()

    ret, frame = cap.read()
    if not ret:
        return jsonify({'error': 'No se pudo capturar la imagen'})

    processed_frame = process_frame(frame, hands, face_mesh)
    encoded_frame = encode_frame(processed_frame)

    cap.release()

    return jsonify({'image': encoded_frame})

if __name__ == '__main__':
    app.run(debug=True)
