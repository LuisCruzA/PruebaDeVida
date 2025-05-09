<<<<<<< HEAD
import cv2 as cv
import pickle as pick
import os
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial.distance import euclidean
import numpy as np

GESTURE_DIR = "gestos_guardados"
os.makedirs(GESTURE_DIR, exist_ok=True)
=======
import cv2
import mediapipe as mp
import base64
import numpy as np
from flask import Flask, jsonify, render_template
>>>>>>> feature/conexion/front-camara

app = Flask(__name__)

<<<<<<< HEAD
def es_ceja_levantada(landmarks):
    ceja = landmarks[55].y
    ojo = landmarks[159].y
    return ceja < ojo - 0.02
=======
# Inicialización de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
>>>>>>> feature/conexion/front-camara

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

<<<<<<< HEAD
    if face_detected.multi_face_landmarks:
        landmarks = face_detected.multi_face_landmarks[0].landmark
        current_face = [(lm.x, lm.y) for lm in landmarks]
        for face_landmarks in face_detected.multi_face_landmarks:
            drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                # mp_face_mesh.FACEMESH_TESSELATION,
                # drawing_styles.get_default_face_mesh_contours_style(),
                # drawing_styles.get_default_face_mesh_tesselation_style(),
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
            )
    
    # key = cv.waitKey(1) & 0xFF
    # if key == ord('s') and face_detected.multi_face_landmarks:
    #     # Presionaste 's': guardar gesto
    #     with open("referencia_gesto.pkl", "wb") as f:
    #         pick.dump(current_face, f)
    #     print("Gesto guardado")
    # elif key == ord('q'):
    #     break

    key = cv.waitKey(1) & 0xFF
    if key == ord('s') and current_face:
        filepath = os.path.join(GESTURE_DIR, f"GestoGuardado.pkl")
        with open(filepath, "wb") as f:
            pick.dump(current_face, f)
        print(f"Gesto guardado en {filepath}")

    elif key == ord('c'):
        filepath = os.path.join(GESTURE_DIR, f"GestoGuardado.pkl")
        with open(filepath, "rb") as f:
            ref_gesture = pick.load(f)

        # Normalizar ambos (por ejemplo, respecto al centro del rostro)
        def normalize(points):
            points = np.array(points)
            center = points.mean(axis=0)
            return points - center

        norm_current = normalize(current_face)
        norm_ref = normalize(ref_gesture)

            # Calcular distancia promedio entre puntos
        dist = np.mean([euclidean(a, b) for a, b in zip(norm_current, norm_ref)])

        if dist < 0.05:  # Umbral ajustable
            print("Gesto reconocido")
                
        elif key == ord('q'):
            break    
        
        

    
    # Display the frame with annotations
    cv.imshow("Show Video", frame)
=======
@app.route("/capture_gestures")
def capture_gestures():
    cap = initialize_camera()
    hands = mp_hands.Hands()
    face_mesh = mp_face_mesh.FaceMesh()

    ret, frame = cap.read()
    if not ret:
        return jsonify({'error': 'No se pudo capturar la imagen'})
>>>>>>> feature/conexion/front-camara

    processed_frame = process_frame(frame, hands, face_mesh)
    encoded_frame = encode_frame(processed_frame)

    cap.release()

    return jsonify({'image': encoded_frame})

if __name__ == '__main__':
    app.run(debug=True)
