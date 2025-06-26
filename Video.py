import cv2
import mediapipe as mp

# --- Inicialización de MediaPipe Hands ---
mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
mp_estilos_dibujo = mp.solutions.drawing_styles

# --- Configuración de MediaPipe Hands ---
manos = mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5)

# --- Captura de video desde la cámara web ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Presiona 'ESC' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignorando fotograma vacío de la cámara.")
        continue

    # Para mejorar el rendimiento, marca la imagen como no escribible (CORRECCIÓN AQUÍ)
    frame.flags.writeable = False
    
    # OpenCV lee en BGR, pero MediaPipe necesita RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen y encontrar las manos
    resultados = manos.process(frame_rgb)

    # Volvemos a hacer la imagen escribible para poder dibujar sobre ella (CORRECCIÓN AQUÍ)
    frame.flags.writeable = True
    
    # Dibujar las anotaciones de la mano en la imagen
    if resultados.multi_hand_landmarks:
        for mano_landmarks in resultados.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(
                frame,
                mano_landmarks,
                mp_manos.HAND_CONNECTIONS,
                mp_estilos_dibujo.get_default_hand_landmarks_style(),
                mp_estilos_dibujo.get_default_hand_connections_style())

            # Opcional: Obtener coordenadas de un punto específico
            alto, ancho, _ = frame.shape
            punta_indice_x = int(mano_landmarks.landmark[mp_manos.HandLandmark.INDEX_FINGER_TIP].x * ancho)
            punta_indice_y = int(mano_landmarks.landmark[mp_manos.HandLandmark.INDEX_FINGER_TIP].y * alto)
            cv2.circle(frame, (punta_indice_x, punta_indice_y), 5, (0, 255, 0), -1)

    # Voltear la imagen horizontalmente para una vista tipo "espejo"
    frame_espejo = cv2.flip(frame, 1)

    # Mostrar el fotograma resultante
    cv2.imshow('Detector de Manos - MediaPipe', frame_espejo)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Liberar recursos
manos.close()
cap.release()
cv2.destroyAllWindows()