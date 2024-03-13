import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import keyboard

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

cap = cv2.VideoCapture(0)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


running = True
hand_points = []  
keyboard_open = False  
mouse_control = False 
while running:
    ret, frame = cap.read() 

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        frame = cv2.flip(frame, 1)  
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))  

      
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                   
                    x = int(landmark.x * WINDOW_WIDTH)
                    y = int(landmark.y * WINDOW_HEIGHT)
                    hand_points.append((x, y))

                if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y:
                    keyboard_open = True
                else:
                    keyboard_open = False

                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y:
                    mouse_control = True
                else:
                    mouse_control = False

       
        if keyboard_open:
            keyboard.press_and_release('ctrl+alt+del') 

   
        if mouse_control:
            mouse_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WINDOW_WIDTH
            mouse_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * WINDOW_HEIGHT
            pyautogui.moveTo(mouse_x, mouse_y)


cap.release()
cv2.destroyAllWindows()
