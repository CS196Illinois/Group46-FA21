import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(1)
time.sleep(1.000)
cap.set(3,640)
cap.set(4,480)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

while(True):
    success, frame = cap.read()
    #frame = cv2.resize(frame, (1024, 768))
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processedHands = hands.process(frame)

    if processedHands.multi_hand_landmarks:
        for point in processedHands.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, processedHands)

    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break