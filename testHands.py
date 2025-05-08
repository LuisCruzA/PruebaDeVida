import cv2 as cv
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import mediapipe.python.solutions.face_mesh as mp_face_mesh

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

    # Display the frame with annotations
    cv.imshow("Show Video", frame)

    # Exit the loop if 'q' key is pressed
    if cv.waitKey(20) & 0xff == ord('q'):
        break

# Release the camera
cam.release()