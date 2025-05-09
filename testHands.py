import cv2 as cv
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from scipy.spatial.distance import euclidean
import math
import numpy as np

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
EYEBROWLEFT = 52
EYELEFT = 159
EYEBROWRIGHT = 33
EYERIGHT = 263

BARBILLA1= 148
BARBILLA2= 152
BARBILLA3= 377

MANDIBULAIZQ = 58
MANDIBULADER = 288

NOSEBOTTOM = 94
NOSETOP = 6

EYELEFTBOTTOM = 145

def distancia_2d(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.sqrt(abs(dx**2 + dy**2))

# def distancia_2d(p1, p2):
#     return math.sqrt((p1.x - p2.x)*2 + (p1.y - p2.y)*2)



def distancia_3d(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    return math.sqrt(dx*2 +dy*2+dz*2)

# Funcion para verificar si la mano es un puño
def is_fist(hand_landmarks):
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    finger_dips = [
        mp_hands.HandLandmark.INDEX_FINGER_DIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
        mp_hands.HandLandmark.RING_FINGER_DIP,
        mp_hands.HandLandmark.PINKY_DIP,
    ]
    for tip, dip in zip(finger_tips, finger_dips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
            return False
    return True

def is_peace(hand_landmarks):
    # Obtener las puntas de los dedos
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Obtener las articulaciones de los dedos
    index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    middle_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinky_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    
    # Verificar si los dedos índice y medio están extendidos y los otros dedos doblados
    if (index_finger_dip.y > index_finger_tip.y and  # Índice extendido
        middle_finger_dip.y > middle_finger_tip.y and  # Medio extendido
        ring_finger_dip.y < ring_finger_tip.y and  # Anular doblado
        pinky_finger_dip.y < pinky_finger_tip.y):  # Meñique doblado
        return True
    return False

def is_thumb_up(hand_landmarks):
    # Obtener las puntas y las articulaciones del pulgar e índice
    thumb_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]  # Articulación base del pulgar
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]  # Articulación índice

    # Comprobar si el pulgar está hacia arriba y los otros dedos están doblados
    if (thumb_finger_tip.y < thumb_finger_mcp.y and  # Pulgar extendido hacia arriba
        thumb_finger_tip.y < index_finger_tip.y and  # Pulgar por encima del índice
        index_finger_dip.y > index_finger_tip.y):    # Índice doblado
        return True
    return False

def is_rock(hand_landmarks):
    # Obtener las puntas y articulaciones de los dedos
    pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]

    # Verificar el gesto "rock" (meñique e índice extendidos, anular y medio doblados)
    if (
        pinky_finger_tip.y < pinky_finger_dip.y and  # Meñique extendido (la punta por debajo de la articulación)
        ring_finger_tip.y > ring_finger_dip.y and  # Anular doblado
        middle_finger_tip.y > middle_finger_dip.y and  # Medio doblado
        index_finger_tip.y < index_finger_dip.y  # Índice extendido (punta por debajo de la articulación)
    ):
        return True
    return False

def is_extended(hand_landmarks):
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.THUMB_TIP
    ]
    finger_dips = [
        mp_hands.HandLandmark.INDEX_FINGER_DIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
        mp_hands.HandLandmark.RING_FINGER_DIP,
        mp_hands.HandLandmark.PINKY_DIP,
        mp_hands.HandLandmark.THUMB_MCP,
    ]
    for tip, dip in zip(finger_tips, finger_dips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[dip].y:
            return False
    return True

# Initialize the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for processing video frames
    max_num_hands=2,           # Maximum number of hands to detect
    min_detection_confidence=0.5  # Minimum confidence threshold for hand detection
)
face = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)
# face_mesh = mp.solutions.face_mesh.FaceMesh()


# Open the camera
cam = cv.VideoCapture(0)

while cam.isOpened():
    # Read a frame from the camera
    success, frame = cam.read()

    # If the frame is not available, skip this iteration
    if not success:
        print("Camera Frame not available")
        continue

    # Convert the frame from BGR to RGB (required by MediaPipe)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    # Process the frame for hand detection and tracking
    hands_detected = hands.process(frame)
    face_detected = face.process(frame)


    # Convert the frame back from RGB to BGR (required by OpenCV)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    gesture_text = ""

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
            def fist():
                if is_fist(hand_landmarks):
                    return("¡Puño cerrado!")

            def peace():
                if is_peace(hand_landmarks):
                    return("¡Paz y amor!")

            def thumb_up():
                if is_thumb_up(hand_landmarks):
                    return("¡Pulgar arriba!")

            def rock():
                if is_rock(hand_landmarks):
                    return("¡Rock!")

            def extended():
                if is_extended(hand_landmarks):
                    return("¡Mano extendida!")

            def GestosdeMano():
                fist()
                peace()
                thumb_up()
                rock()
                extended()
            GestosdeMano()

    if face_detected.multi_face_landmarks:
        landmarks = face_detected.multi_face_landmarks[0].landmark
        current_face = [(lm.x, lm.y) for lm in landmarks]
        for face_landmarks in face_detected.multi_face_landmarks:
            # drawing.draw_landmarks(
            #     frame,
            #     # face_landmarks,
            #     # mp_face_mesh.FACEMESH_CONTOURS,
            #     # mp_face_mesh.FACEMESH_TESSELATION,
            #     # drawing_styles.get_default_face_mesh_contours_style(),
            #     # drawing_styles.get_default_face_mesh_tesselation_style(),
            #     # landmark_drawing_spec=None,
            #     # connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
            # )

            left = landmarks[MOUTH_LEFT]
            right = landmarks[MOUTH_RIGHT]
            top = landmarks[MOUTH_TOP]
            bottom = landmarks[MOUTH_BOTTOM]

            eyeleft = landmarks[EYELEFT]
            eyebrowleft = landmarks[EYEBROWLEFT]
            eyeright = landmarks[EYERIGHT]
            eyebrowright = landmarks[EYEBROWRIGHT]

            barbilla1 = landmarks[BARBILLA1]
            barbilla2 = landmarks[BARBILLA2]
            barbilla3 = landmarks[BARBILLA3]

            mandibulaIzq = landmarks[MANDIBULAIZQ]
            mandibulaDer = landmarks[MANDIBULADER]

            nosetop = landmarks[NOSETOP]
            nosebottom = landmarks[NOSEBOTTOM]

            eyeleftbottom = landmarks[EYELEFTBOTTOM]

            # Convertir a píxeles
            left_pt = (int(left.x * w), int(left.y * h))
            right_pt = (int(right.x * w), int(right.y * h))
            top_pt = (int(top.x * w), int(top.y * h))
            bottom_pt = (int(bottom.x * w), int(bottom.y * h))

            eyeleft_pt = (int(eyeleft.x * w), int(eyeleft.y * h))
            eyebrowleft_pt = (int(eyebrowleft.x * w), int(eyebrowleft.y * h))
            eyeright_pt = (int(eyeright.x * w), int(eyeright.y * h))
            eyebrowright_pt = (int(eyebrowright.x * w), int(eyebrowright.y * h))

            barbilla1_pt = (int(barbilla1.x * w), int(barbilla1.y * h))
            barbilla2_pt = (int(barbilla2.x * w), int(barbilla2.y * h))
            barbilla3_pt = (int(barbilla3.x * w), int(barbilla3.y * h))

            mandibulaIzq_pt = (int(mandibulaIzq.x * w), int(mandibulaIzq.y * h))
            mandibulaDer_pt = (int(mandibulaDer.x * w), int(mandibulaDer.y * h))

            nosebottom_pt = (int(nosebottom.x * w), int(nosebottom.y * h))
            nosetop_pt = (int(nosetop.x * w), int(nosetop.y * h))

            eyeleftbottom_pt = (int(eyeleftbottom.x * w), int(eyeleftbottom.y * h))

            # Dibujar puntos
            for pt in [left_pt, right_pt, top_pt, bottom_pt]:
                cv.circle(frame, pt, 2, (0, 255, 0), -1)

            for pt in [eyeleft_pt, eyebrowleft_pt, eyeright_pt, eyebrowright_pt, eyeleftbottom_pt]:
                cv.circle(frame, pt, 2, (0, 255, 0), -1)
            
            for pt in [barbilla1_pt, barbilla2_pt, barbilla3_pt]:
                cv.circle(frame, pt, 2, (0, 255, 0), -1)
            
            for pt in [mandibulaIzq_pt, mandibulaDer_pt]:
                cv.circle(frame, pt, 2, (0, 255, 0), -1)

            for pt in [nosebottom_pt, nosetop_pt]:
                cv.circle(frame, pt, 2, (0, 255, 0), -1)    

            # Calcular distancias
            mouth_width = euclidean(left_pt, right_pt)
            mouth_height = euclidean(top_pt, bottom_pt)

            # Relación: ancho vs. alto de la boca
            ratio = mouth_width / mouth_height if mouth_height != 0 else 0

            distance_eye_eyebrow = distancia_2d(eyeleft, eyebrowleft)

            # Distancia entre pupilas (usada para normalizar)
            dist_pupilas = distancia_2d(eyeright, eyeleft)

            distance_left_head = distancia_2d(left, mandibulaIzq)
            distance_right_head = distancia_2d(mandibulaDer, right)
            dist_nose = distancia_2d(nosebottom, nosetop)

            dist_eye = distancia_2d(eyeleft, eyeleftbottom)

            # Distancia normalizada
            dist_normalizada = distance_eye_eyebrow / dist_pupilas
            dist_normalizada_ojos = dist_eye / dist_pupilas
            # print(dist_normalizada_ojos)

            dist_norm_headright = distance_left_head / dist_nose

            def Sonrisa():
                if ratio > 10:
                    # cv.putText(frame, "Neutral", (30, 50),
                    # cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    return False
                else:
                    # cv.putText(frame, "¡Sonriendo!", (30, 50),
                    # cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    return("Está sonriendo") 

            def CejaLevantada():
                if dist_normalizada > 0.65:
                    # cv.putText(frame, "Ceja levantada! ", (30, 50),
                    # cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    return( "Está levantando una ceja")
                else:
                    # cv.putText(frame, "Neutral", (30, 50),
                    # cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    return False
                    
            def CaradeLado():
                if distance_left_head > dist_nose:
                    # cv.putText(frame, "Cara de Lado! ", (30, 50),
                    # cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    return( "Está girando la cabeza")
                elif distance_right_head > dist_nose:
                    # cv.putText(frame, "Está girando la cabeza ", (30, 50),
                    # cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    return( "Está girando la cabeza")
                else:
                    # cv.putText(frame, "Neutral", (30, 50),
                    # cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    return False
                    
            def BocaAbierta():
                if ratio > 1 and ratio <4:
                    # cv.putText(frame, "Boca Abierta", (30, 50),
                    # cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    return( "Está abriendo la boca")
                else:
                    # cv.putText(frame, "Neutral", (30, 50),
                    # cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    return False
                
            def ReconocerGestos():
                Sonrisa()
                CejaLevantada()
                CaradeLado()
                BocaAbierta()
                

            ReconocerGestos()

    # Display the frame with annotations
    cv.imshow("Show Video", frame)

    # Exit the loop if 'q' key is pressed
    if cv.waitKey(20) & 0xff == ord('q'):
        break

# Release the camera
cam.release()
