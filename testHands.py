import cv2
import pickle as pick
import os
import mediapipe as mp
from scipy.spatial.distance import euclidean
import numpy as np
from random import choice
from flask import Flask, render_template, Response

GESTURE_DIR = "gestos_guardados"
os.makedirs(GESTURE_DIR, exist_ok=True)

app = Flask(__name__)

# Inicializar modelos de MediaPipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)

gestos = ["Sonrisa"]
gestoAleatorio = choice(gestos)
print(f"Gesto aleatorio seleccionado: {gestoAleatorio}")

# Función para verificar si está sonriendo
def is_smiling(landmarks):
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    top_lip = landmarks[0]
    mouth_width = abs(left_mouth.x - right_mouth.x)
    mouth_height = abs((left_mouth.y + right_mouth.y) / 2 - top_lip.y)
    return mouth_width > 0.25 and mouth_height > 0.05

# Normalizar puntos para comparar gestos
def normalize(points):
    points = np.array(points)
    center = points.mean(axis=0)
    return points - center

def gen_frames():
    cam = cv2.VideoCapture(0)
    current_face = None

    while True:
        success, frame = cam.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_detected = hands.process(frame_rgb)
        face_detected = face.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if hands_detected.multi_hand_landmarks:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        if face_detected.multi_face_landmarks:
            landmarks = face_detected.multi_face_landmarks[0].landmark
            current_face = [(lm.x, lm.y) for lm in landmarks]

            for face_landmarks in face_detected.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

            if is_smiling(landmarks):
                cv2.putText(frame_bgr, "¡Está sonriendo!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        key = cv2.waitKey(1) & 0xFF

        # Guardar gesto
        if key == ord('s') and current_face:
            filepath = os.path.join(GESTURE_DIR, f"Sonrisa.pkl")
            with open(filepath, "wb") as f:
                pick.dump(current_face, f)
            print(f"Gesto guardado en {filepath}")

        # Comparar gesto
        elif key == ord('c') and current_face:
            filepath = os.path.join(GESTURE_DIR, f"{gestoAleatorio}.pkl")
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    ref_gesture = pick.load(f)

                norm_current = normalize(current_face)
                norm_ref = normalize(ref_gesture)

                dist = np.mean([euclidean(a, b) for a, b in zip(norm_current, norm_ref)])
                if dist < 0.05:
                    cv2.putText(frame_bgr, "Gesto reconocido", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    print("Gesto reconocido")

        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cam.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
