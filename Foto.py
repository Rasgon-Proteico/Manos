import cv2
import mediapipe as mp

# --- Inicialización ---
mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
    
# --- Cargar la imagen ---
ruta_imagen = 'jala.jpg'  # <-- CAMBIA ESTO por la ruta de tu imagen
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print(f"Error: No se pudo cargar la imagen en '{ruta_imagen}'")
else:
    with mp_manos.Hands(
        static_image_mode=True, # Ideal para imágenes estáticas
        max_num_hands=2,
        min_detection_confidence=0.5) as manos:
        
        # Convertir la imagen BGR a RGB
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen
        resultados = manos.process(imagen_rgb)

        # Dibujar los landmarks si se detectaron manos
        if resultados.multi_hand_landmarks:
            print(f"Se encontraron {len(resultados.multi_hand_landmarks)} mano(s).")
            for mano_landmarks in resultados.multi_hand_landmarks:
                mp_dibujo.draw_landmarks(
                    imagen,
                    mano_landmarks,
                    mp_manos.HAND_CONNECTIONS)
            
            # Guardar o mostrar la imagen con los landmarks
            cv2.imwrite('resultado_mano.png', imagen)
            print("Imagen guardada como 'resultado_mano.png'")

            # Mostrar la imagen
            cv2.imshow('Mano Detectada', imagen)
            cv2.waitKey(0) # Espera a que se presione una tecla
            cv2.destroyAllWindows()
        else:
            print("No se encontraron manos en la imagen.")