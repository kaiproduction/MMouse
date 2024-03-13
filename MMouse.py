import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import keyboard  # Импорт библиотеки keyboard

# Размеры окна Pygame
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)

# Инициализация Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Основной цикл программы
running = True
hand_points = []  # Список для хранения координат точек на руках
keyboard_open = False  # Флаг для открытия клавиатуры
mouse_control = False  # Флаг для управления мышью
while running:
    ret, frame = cap.read()  # Захват кадра с камеры

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Конвертация цветового пространства
        frame = cv2.flip(frame, 1)  # Зеркальное отображение кадра
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))  # Изменение размера кадра

        # Поиск ключевых точек на руках
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    # Получение координат ключевых точек
                    x = int(landmark.x * WINDOW_WIDTH)
                    y = int(landmark.y * WINDOW_HEIGHT)
                    hand_points.append((x, y))

                # Определение жеста для открытия клавиатуры
                if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y:
                    keyboard_open = True
                else:
                    keyboard_open = False

                # Определение жеста для управления мышью
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y:
                    mouse_control = True
                else:
                    mouse_control = False

        # Если жест для открытия клавиатуры обнаружен, открываем клавиатуру
        if keyboard_open:
            keyboard.press_and_release('ctrl+alt+del')  # Пример нажатия сочетания клавиш для открытия клавиатуры

        # Если жест для управления мышью обнаружен, симулируем движение мыши
        if mouse_control:
            mouse_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WINDOW_WIDTH
            mouse_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * WINDOW_HEIGHT
            pyautogui.moveTo(mouse_x, mouse_y)

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
