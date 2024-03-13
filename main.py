import cv2
import pygame
import numpy as np
import mediapipe as mp

# Инициализация Pygame
pygame.init()

# Размеры окна Pygame
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Инициализация окна Pygame
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Video Display")

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)

# Инициализация Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Основной цикл программы
running = True
hand_points = []  # Список для хранения координат точек на руках
wearing_headphones = False  # Флаг для определения, надеты ли наушники
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

                # Проверка наличия наушников
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y:
                    wearing_headphones = True
                else:
                    wearing_headphones = False

        # Отображение точек на экране
        screen.fill(BLACK)  # Очистка экрана
        for point in hand_points:
            pygame.draw.circle(screen, WHITE, point, 5)  # Отрисовка точек
        pygame.display.flip()  # Обновление экрана

        hand_points = []  # Очистка списка точек перед следующим кадром

    # Обработка событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
pygame.quit()
