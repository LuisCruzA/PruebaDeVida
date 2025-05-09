import cv2
import mediapipe as mp
from flask import Flask, render_template, Response
import threading

app = Flask(__name__)

# Inicializar los modelos de MediaPipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración del modelo de manos
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Configuración del modelo de rostros
face = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)

# Abre la cámara
cam = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cam.read()
        if not success:
            break

        # Convierte el frame a RGB (MediaPipe requiere RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesa el frame para detectar las manos y el rostro
        hands_detected = hands.process(frame_rgb)
        face_detected = face.process(frame_rgb)

        # Convierte el frame de nuevo a BGR (OpenCV requiere BGR)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Dibuja los puntos de la mano si se detectan
        if hands_detected.multi_hand_landmarks:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        # Dibuja los puntos del rostro si se detectan
        if face_detected.multi_face_landmarks:
            for face_landmarks in face_detected.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        # Codifica el frame en formato JPEG
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        if not ret:
            break

        # Convierte el frame a bytes
        frame_bytes = buffer.tobytes()

        # Envía el frame como parte del flujo de imágenes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
