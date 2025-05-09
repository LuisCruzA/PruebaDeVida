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
from random import *

GESTURE_DIR = "gestos_guardados"
os.makedirs(GESTURE_DIR, exist_ok=True)

# Initialize the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for processing video frames
    max_num_hands=2,           # Maximum number of hands to detect
    min_detection_confidence=0.5  # Minimum confidence threshold for hand detection
)
face = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    min_detection_confidence=0.5)
# face_mesh = mp.solutions.face_mesh.FaceMesh()


# Open the camera
cam = cv.VideoCapture(0)
gestos = ["Sonrisa"]
gestoAleatorio = choice(gestos)
print(gestoAleatorio)



def is_smiling(landmarks):
        left_mouth = landmarks[61]  # Comisura izquierda
        right_mouth = landmarks[291]  # Comisura derecha
        top_lip = landmarks[0]  # Nariz base
        mouth_width = abs(left_mouth.x - right_mouth.x)
        mouth_height = abs((left_mouth.y + right_mouth.y) / 2 - top_lip.y)
        
        return mouth_width > 0.25 and mouth_height>0.05

while cam.isOpened():
    # Read a frame from the camera
    success, frame = cam.read()

    # If the frame is not available, skip this iteration
    if not success:
        print("Camera Frame not available")
        continue

    # Convert the frame from BGR to RGB (required by MediaPipe)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame for hand detection and tracking
    hands_detected = hands.process(frame)
    face_detected = face.process(frame)


    # Convert the frame back from RGB to BGR (required by OpenCV)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # If hands are detected, draw landmarks and connections on the frame
    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

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
        if is_smiling(face_landmarks.landmark):
                print("¡Está sonriendo!")    
    
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
        filepath = os.path.join(GESTURE_DIR, f"Sonrisa.pkl")
        with open(filepath, "wb") as f:
            pick.dump(current_face, f)
        print(f"Gesto guardado en {filepath}")

    elif key == ord('c'):
        
        filepath = os.path.join(GESTURE_DIR, f"{gestoAleatorio}.pkl")
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

    # Exit the loop if 'q' key is pressed
    if cv.waitKey(20) & 0xff == ord('q'):
        break

# Release the camera
cam.release()