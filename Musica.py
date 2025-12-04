import cv2
import pygame
import numpy as np
import time

# Inicializar pygame mixer
pygame.mixer.init()

# Cargar sonidos
notes = {}
try:
    notes = {
        'C': pygame.mixer.Sound("C.wav"),
        'D': pygame.mixer.Sound("D.wav"),
        'E': pygame.mixer.Sound("E.wav"),
        'F': pygame.mixer.Sound("F.wav"),
        'G': pygame.mixer.Sound("G.wav"),
        'A': pygame.mixer.Sound("A.wav"),
        'B': pygame.mixer.Sound("B.wav"),
        'C1': pygame.mixer.Sound("C1.wav")
    }
except:
    print("Error cargando sonidos. Creando sonidos de prueba...")
    # Crear sonidos de piano más realistas
    base_freq = 261.63  # Do central 
    frequencies = {
        'C': base_freq,
        'D': base_freq * (9/8),
        'E': base_freq * (5/4),
        'F': base_freq * (4/3),
        'G': base_freq * (3/2),
        'A': base_freq * (5/3),
        'B': base_freq * (15/8),
        'C1': base_freq * 2
    }
    
    for note, freq in frequencies.items():
        duration = 1.5
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Onda más rica con armónicos
        wave = (np.sin(2 * np.pi * freq * t) * 0.5 +
               np.sin(2 * np.pi * freq * 2 * t) * 0.3 +
               np.sin(2 * np.pi * freq * 3 * t) * 0.15 +
               np.sin(2 * np.pi * freq * 4 * t) * 0.05)
        
        # Envolvente para sonido natural
        envelope = np.ones_like(t)
        attack = int(0.1 * sample_rate)
        decay = int(0.2 * sample_rate)
        release = int(0.5 * sample_rate)
        
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack+decay] = np.linspace(1, 0.7, decay)
        envelope[-release:] = np.linspace(0.7, 0, release)
        
        wave *= envelope
        wave = (wave * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(wave)
        notes[note] = sound

finger_states = {note: False for note in notes.keys()}
last_play_time = {note: 0 for note in notes.keys()}
cooldown = 0.3  # Segundos entre repeticiones de la misma nota

# Variables para detección de movimiento
previous_frame = None
motion_history = None

# Configuración de teclas
KEY_HEIGHT = 120  # Altura de las teclas 
KEY_POSITION = "top"  # "top" o "bottom" - posición de las teclas

def detect_skin_color(frame):
    """Detectar color de piel en el frame"""
    # Convertir a HSV para mejor detección de piel
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Rangos para color de piel (ajustables según iluminación)
    lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
    lower_skin2 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([20, 255, 255], dtype=np.uint8)
    
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    # Operaciones morfológicas para limpiar la máscara
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    return skin_mask

def detect_motion(frame):
    """Detectar movimiento entre frames"""
    global previous_frame, motion_history
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if previous_frame is None:
        previous_frame = gray
        motion_history = np.zeros_like(gray)
        return np.zeros_like(gray)
    
    # Calcular diferencia entre frames
    frame_diff = cv2.absdiff(previous_frame, gray)
    _, motion_mask = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
    
    # Actualizar historial de movimiento
    if motion_history is not None:
        motion_history = cv2.addWeighted(motion_history, 0.7, motion_mask, 0.3, 0)
    
    previous_frame = gray
    return motion_mask

def detect_presence_in_regions(frame):
    """Detectar presencia en las regiones usando combinación de técnicas"""
    # Obtener máscaras de piel y movimiento
    skin_mask = detect_skin_color(frame)
    motion_mask = detect_motion(frame)
    
    # Combinar ambas detecciones
    combined_mask = cv2.bitwise_or(skin_mask, motion_mask)
    
    # Mejorar la máscara combinada
    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Dividir en regiones según la posición de las teclas
    height, width = frame.shape[:2]
    regions = []
    
    for i in range(8):
        x_start = i * (width // 8)
        x_end = (i + 1) * (width // 8)
        
        if KEY_POSITION == "top":
            # Teclas en la parte superior
            region = combined_mask[0:KEY_HEIGHT, x_start:x_end]
        else:
            # Teclas en la parte inferior
            region = combined_mask[height-KEY_HEIGHT:height, x_start:x_end]
        
        regions.append(region)
    
    return regions

def play_notes_based_on_presence(regions):
    """Reproducir notas basado en la presencia en las regiones"""
    note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C1']
    current_time = time.time()
    
    for i, region in enumerate(regions):
        note_name = note_names[i]
        
        # Calcular presencia en la región
        presence_level = cv2.countNonZero(region)
        
        # Umbral más bajo para detección más sensible
        if presence_level > 200:  # Umbral reducido para mayor sensibilidad
            if (not finger_states[note_name] and 
                current_time - last_play_time[note_name] > cooldown):
                
                notes[note_name].play()
                finger_states[note_name] = True
                last_play_time[note_name] = current_time
        else:
            finger_states[note_name] = False

def draw_compact_interface(frame, regions):
    """Dibujar interfaz compacta con teclas en los bordes"""
    height, width = frame.shape[:2]
    note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C1']
    
    # Colores vibrantes para las teclas
    key_colors = [
        (255, 100, 100),    # Rojo
        (255, 150, 50),     # Naranja
        (255, 200, 50),     # Amarillo
        (150, 255, 100),    # Verde claro
        (50, 255, 150),     # Verde
        (50, 200, 255),     # Azul claro
        (100, 150, 255),    # Azul
        (200, 100, 255)     # Violeta
    ]
    
    # Determinar posición Y de las teclas
    if KEY_POSITION == "top":
        y_start = 0
        y_end = KEY_HEIGHT
        info_y = KEY_HEIGHT + 30
    else:
        y_start = height - KEY_HEIGHT
        y_end = height
        info_y = height - KEY_HEIGHT - 30
    
    for i in range(8):
        x_start = i * (width // 8)
        x_end = (i + 1) * (width // 8)
        
        # Calcular nivel de presencia
        presence_level = cv2.countNonZero(regions[i])
        presence_ratio = min(presence_level / 800, 1.0)
        
        # Color de la tecla basado en presencia
        if finger_states[note_names[i]]:
            # Tecla activa - color brillante
            color = key_colors[i]
            border_color = (0, 255, 0)
            border_thickness = 4
        else:
            # Tecla inactiva - color atenuado por presencia
            base_color = key_colors[i]
            color = tuple(int(c * (0.4 + 0.6 * presence_ratio)) for c in base_color)
            border_color = (80, 80, 80)
            border_thickness = 2
        
        # Dibujar tecla con relleno
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, -1)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), border_color, border_thickness)
        
        # Nombre de la nota 
        text_x = x_start + (width // 8) // 2 - 15
        text_y = y_start + KEY_HEIGHT // 2 + 10
        
        cv2.putText(frame, note_names[i], 
                   (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Barra de nivel de presencia pequeña dentro de la tecla
        bar_width = int(presence_ratio * (width // 8 - 20))
        if KEY_POSITION == "top":
            bar_y = y_end - 15
        else:
            bar_y = y_start + 15
            
        cv2.rectangle(frame, 
                     (x_start + 10, bar_y - 5), 
                     (x_start + 10 + bar_width, bar_y + 5), 
                     (0, 255, 0), -1)
        
        # Punto indicador de actividad
        if finger_states[note_names[i]]:
            dot_color = (0, 255, 0)
        else:
            dot_color = (100, 100, 100)
            
        dot_x = x_start + (width // 8) // 2
        if KEY_POSITION == "top":
            dot_y = y_end - 30
        else:
            dot_y = y_start + 30
            
        cv2.circle(frame, (dot_x, dot_y), 8, dot_color, -1)

    # Mostrar información de detección
    total_presence = sum([cv2.countNonZero(region) for region in regions])
    cv2.putText(frame, f"Deteccion: {total_presence}", 
               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Línea separadora
    if KEY_POSITION == "top":
        cv2.line(frame, (0, KEY_HEIGHT), (width, KEY_HEIGHT), (100, 100, 100), 2)
    else:
        cv2.line(frame, (0, height-KEY_HEIGHT), (width, height-KEY_HEIGHT), (100, 100, 100), 2)

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Configurar cámara para mejor rendimiento
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print(" Piano Virtual - Teclas Compactas")
print("===========================================")
print(f"Teclas en la parte {KEY_POSITION}")
print(f"Altura de teclas: {KEY_HEIGHT}px")
print("Instrucciones:")
print("- Coloca tu mano sobre las teclas para tocarlas")
print("- Las barras verdes muestran el nivel de detección")
print("- Presiona 'q' para salir")
print("- Presiona 't' para cambiar posición de teclas (arriba/abajo)")
print("- Presiona 'r' para resetear el detector")
print("===========================================")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede leer la cámara")
        break
    
    # Voltear frame para modo espejo
    frame = cv2.flip(frame, 1)
    
    # Detectar presencia en regiones
    regions = detect_presence_in_regions(frame)
    
    # Reproducir notas basado en presencia
    play_notes_based_on_presence(regions)
    
    # Dibujar interfaz compacta
    draw_compact_interface(frame, regions)
    
    # Mostrar información de ayuda
    help_text = f"Teclas {KEY_POSITION} - 't' para cambiar - 'q' salir"
    cv2.putText(frame, help_text, 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Piano Virtual - Teclas Compactas', frame)
    
    # Controles
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset del detector
        previous_frame = None
        motion_history = None
        print("Detector reseteado")
    elif key == ord('t'):  # Cambiar posición de teclas
        KEY_POSITION = "bottom" if KEY_POSITION == "top" else "top"
        print(f"Teclas movidas a la parte {KEY_POSITION}")

cap.release()
cv2.destroyAllWindows()
pygame.quit()