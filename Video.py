import cv2
import mediapipe as mp
import pygame

# --- Inicialización de MediaPipe Hands ---
mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
mp_estilos_dibujo = mp.solutions.drawing_styles

# --- Configuración de Pygame ---
pygame.init()
sonidos = [
    pygame.mixer.Sound('Sounds/do.wav'),    #?0 - Mano 1, Dedo 1 (Índice)
    pygame.mixer.Sound('Sounds/re.wav'),    #?1 - Mano 1, Dedo 2 (Medio)
    pygame.mixer.Sound('Sounds/mi.wav'),    #?2 - Mano 1, Dedo 3 (Anular)
    pygame.mixer.Sound('Sounds/fa.wav'),    #?3 - (No se usa con la lógica actual)
    pygame.mixer.Sound('Sounds/sol.wav'),   #?4 - Mano 2, Dedo 1 (Índice)
    pygame.mixer.Sound('Sounds/la.wav'),    #?5 - Mano 2, Dedo 2 (Medio)
    pygame.mixer.Sound('Sounds/si.wav'),    #?6 - Mano 2, Dedo 3 (Anular)
]

# --- CORRECCIÓN 1: Lógica de la función is_finger_down ---
# La función ahora recibe la lista de landmarks y los ÍNDICES de la punta y la base del dedo.
def is_finger_down(lista_landmarks, tip_index, mcp_index):
    # Compara la coordenada Y de la punta (tip) con la de la articulación principal (mcp).
    # En OpenCV, el eje Y aumenta hacia abajo, por lo que si la punta tiene un valor Y mayor, está "doblado" o "abajo".
    return lista_landmarks[tip_index].y > lista_landmarks[mcp_index].y

manos = mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5)

# Usaremos 7 estados, uno para cada sonido/dedo posible.
finger_state = [False] * 7

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

    # Voltear la imagen horizontalmente para una vista tipo "espejo" y mejorar la interacción.
    frame = cv2.flip(frame, 1)

    # Para mejorar el rendimiento, marca la imagen como no escribible
    frame.flags.writeable = False
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen y encontrar las manos
    resultados = manos.process(frame_rgb)

    # Volvemos a hacer la imagen escribible para poder dibujar sobre ella
    frame.flags.writeable = True

    if resultados.multi_hand_landmarks:
        # Iteramos sobre cada mano detectada (m será 0 para la primera mano, 1 para la segunda)
        for m, mano_landmarks in enumerate(resultados.multi_hand_landmarks):
            # --- CORRECCIÓN 2: Definición de listas y bucle fuera de la función de dibujo ---
            # Usamos las constantes de MediaPipe en lugar de números para más claridad.
            finger_tips_indices = [mp_manos.HandLandmark.INDEX_FINGER_TIP, mp_manos.HandLandmark.MIDDLE_FINGER_TIP, mp_manos.HandLandmark.RING_FINGER_TIP]
            finger_mcp_indices = [mp_manos.HandLandmark.INDEX_FINGER_MCP, mp_manos.HandLandmark.MIDDLE_FINGER_MCP, mp_manos.HandLandmark.RING_FINGER_MCP]

            # Iteramos sobre los 3 dedos que queremos comprobar (índice, medio, anular)
            for i in range(3):
                # --- CORRECCIÓN 3: Cálculo del índice para sonidos y estado ---
                # Mano 0 (m=0): i=0,1,2 -> finger_index = 0,1,2
                # Mano 1 (m=1): i=0,1,2 -> finger_index = 4,5,6
                # Esta lógica salta el índice 3, tal como en tu código original.
                finger_index = i + (m * 4)

                # --- CORRECCIÓN 4: Llamada correcta a la función is_finger_down ---
                # Pasamos la lista de landmarks de la mano actual (mano_landmarks.landmark) y los índices del dedo.
                if is_finger_down(mano_landmarks.landmark, finger_tips_indices[i], finger_mcp_indices[i]):
                    # Si el dedo está ABAJO y su estado anterior era ARRIBA...
                    if not finger_state[finger_index]:
                        print(f"Tocando sonido {finger_index}") # Ayuda para depurar
                        sonidos[finger_index].play()  # ...reproducir sonido.
                        finger_state[finger_index] = True  # Actualizar el estado del dedo a "ABAJO".
                else:
                    # Si el dedo está ARRIBA, reseteamos su estado.
                    finger_state[finger_index] = False

            # --- CORRECCIÓN 5: Reactivar el dibujo de las manos ---
            # Es útil para ver que la detección funciona correctamente.
            mp_dibujo.draw_landmarks(
                frame,
                mano_landmarks,
                mp_manos.HAND_CONNECTIONS,
                mp_estilos_dibujo.get_default_hand_landmarks_style(),
                mp_estilos_dibujo.get_default_hand_connections_style())

    # Mostrar el fotograma resultante
    cv2.imshow('Piano Virtual - MediaPipe', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Liberar recursos
manos.close()
cap.release()
cv2.destroyAllWindows()
pygame.quit()