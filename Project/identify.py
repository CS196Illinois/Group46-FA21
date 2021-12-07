import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import time

modelFile = "./models/rhand_alphabet_marcelh2_lr.pkl"

# Initialize model
with open(modelFile, 'rb') as f:
    model = pickle.load(f)

# Start 640x480 resolution camera capture
cap = cv2.VideoCapture(1)
time.sleep(1.000)
cap.set(3,640)
cap.set(4,480)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while(True):
    success, frame = cap.read()

    # Handle keyboard inputs
    keypress = cv2.waitKey(1)

    # q means exit program
    if keypress & 0xFF == ord('q'):
        break

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

            # Generate coordinate array to pass into model pipeline
            handRow = list(np.array([[lm.x, lm.y, lm.z] for lm in handLms.landmark]).flatten())

            X = pd.DataFrame([handRow])
            displayDefinition = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            # TODO: Fix probabilities: https://towardsdatascience.com/pythons-predict-proba-doesn-t-actually-predict-probabilities-and-how-to-fix-it-f582c21d63fc

            cv2.putText(frame, text = displayDefinition + " p=" + str(probability), org = (0, 100), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 0), thickness = 2, lineType = cv2.LINE_AA)
            mpDraw.draw_landmarks(frame, handLms)

    # Render each frame
    cv2.imshow("Identify Test", frame)