import cv2
import mediapipe as mp
from scipy.spatial.distance import euclidean
# import mediapipe.python.solutions.drawing_styles as drawing_styles
import math


# Inicializa MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)
drawing = mp.solutions.drawing_utils

# Índices de landmarks de la boca
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

cap = cv2.VideoCapture(0)

def es_ceja_levantada(landmarks):
    ceja = landmarks[55].y
    ojo = landmarks[159].y
    return ceja < ojo - 0.02

def distancia_2d(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.sqrt(abs(dx*2 + dy*2))

# def distancia_2d(p1, p2):
#     return math.sqrt((p1.x - p2.x)*2 + (p1.y - p2.y)*2)



def distancia_3d(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    return math.sqrt(dx*2 +dy*2+dz*2)

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

parpadeos = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

            
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Coordenadas clave
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
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            for pt in [eyeleft_pt, eyebrowleft_pt, eyeright_pt, eyebrowright_pt, eyeleftbottom_pt]:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)
            
            for pt in [barbilla1_pt, barbilla2_pt, barbilla3_pt]:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)
            
            for pt in [mandibulaIzq_pt, mandibulaDer_pt]:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            for pt in [nosebottom_pt, nosetop_pt]:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)    

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
            print(dist_normalizada_ojos)

            dist_norm_headright = distance_left_head / dist_nose


            # ESTO SÍ SIRVE
            # Detectar sonrisa si la boca está más ancha de lo normal
            # if ratio > 10:
            #     cv2.putText(frame, "Neutral", (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # else:
            #     cv2.putText(frame, "¡Sonriendo!", (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
            # if dist_normalizada > 0.65:
            #     cv2.putText(frame, "Ceja levantada! ", (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # else:
            #     cv2.putText(frame, "Neutral", (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # if distance_left_head > dist_nose:
            #     cv2.putText(frame, "CCara de Lado! ", (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # else:
            #     cv2.putText(frame, "Neutral", (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
            # if distance_right_head > dist_nose:
            #     cv2.putText(frame, "CCara de Lado! ", (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # else:
            #     cv2.putText(frame, "Neutral", (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # if ratio > 1 and ratio <4:
            #     cv2.putText(frame, "Boca Abierta", (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # else:
            #     cv2.putText(frame, "Neutral", (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, "Parpadeos: " + str(parpadeos), (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            if dist_normalizada_ojos < 0.15:
                parpadeos+1
            else:
                cv2.putText(frame, "Neutral", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                

    cv2.imshow("Detector de sonrisa", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
