import cv2
import mediapipe as mp
import time

# Start 640x480 resolution camera capture
cap = cv2.VideoCapture(1)
time.sleep(1.000)
cap.set(3,640)
cap.set(4,480)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while(True):
    success, frame = cap.read()
    #frame = cv2.resize(frame, (1024, 768))
    # Mirror camera so handedness is correct
    frame = cv2.flip(frame, 1)
    processedHands = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Frame information
    h, w, c = frame.shape
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h

    # Draw hand landmarks on each frame
    if processedHands.multi_hand_landmarks:
        for idx, handLms in enumerate(processedHands.multi_hand_landmarks):
            lbl = processedHands.multi_handedness[idx].classification[0].label
            # bounding box code: https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d
            for lm in handLms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, text = lbl, org = (x_min, y_min), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 0), thickness = 2, lineType = cv2.LINE_AA)
            mpDraw.draw_landmarks(frame, handLms)

    # Render each frame
    cv2.imshow("Vision Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break