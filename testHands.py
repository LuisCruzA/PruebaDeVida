import cv2 as cv
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import mediapipe.python.solutions.face_mesh as mp_face_mesh

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
            if is_fist(hand_landmarks):
                print("¡Puño cerrado!")

            if is_peace(hand_landmarks):
                print("¡Paz y amor!")

            if is_thumb_up(hand_landmarks):
                print("¡Pulgar arriba!")

            if is_thumb_up(hand_landmarks):
                print("¡Pulgar arriba!")

            if is_rock(hand_landmarks):
                print("¡Rock!")

            if is_extended(hand_landmarks):
                print("¡Mano extendida!")

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

    # Display the frame with annotations
    cv.imshow("Show Video", frame)

    # Exit the loop if 'q' key is pressed
    if cv.waitKey(20) & 0xff == ord('q'):
        break

# Release the camera
cam.release()
