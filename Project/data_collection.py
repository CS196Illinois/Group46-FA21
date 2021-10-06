import csv
import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
import os

datasetFile = "./datasets/thumbs.csv"
imageDefinition = "Enter image definition by pressing 'd'"
imageTaken = False

# Start 640x480 resolution camera capture
cap = cv2.VideoCapture(1)
time.sleep(1.000)
cap.set(3,640)
cap.set(4,480)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialize CSV columns and file if necessary
headers = ['definition']
for idx in range(0, 21): # 21 hand landmark coordinates
    headers += ['x{}'.format(idx),'y{}'.format(idx),'z{}'.format(idx)]
with open(datasetFile, mode='a', newline='') as f:
    if os.path.getsize(datasetFile) == 0:
        print("[*] Initializing new dataset in file {}".format(datasetFile))
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(headers)
    else:
        print("[*] Appending to current dataset in file {}".format(datasetFile))

while(True):
    success, frame = cap.read()

    # Handle keyboard inputs
    keypress = cv2.waitKey(1)

    # q means exit program
    if keypress & 0xFF == ord('q'):
        break
    # space means start recording definition
    elif keypress & 0xFF == ord(' '):
        imageTaken = True
        imageToProcess = frame
        print("[*] Started recording")
    # s means stop recording definition
    elif keypress & 0xFF == ord('s'):
        imageTaken = False
        print("[*] Stopped recording")
    # d means define image definition
    elif keypress & 0xFF == ord('d'):
        tk_root = tk.Tk()
        tk_root.withdraw()
        imageDefinition = tk.simpledialog.askstring(title="Train", prompt="Enter image definition:")

    #frame = cv2.resize(frame, (1024, 768))
    # Mirror camera so handedness is correct
    frame = cv2.flip(frame, 1)
    processedHands = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Display current image definition
    cv2.putText(frame, text = imageDefinition, org = (0, 100), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255, 0, 0), thickness = 2)

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

    # Save hand landmarks to CSV file
    if processedHands.multi_hand_landmarks and imageTaken:
        for idx, handLms in enumerate(processedHands.multi_hand_landmarks):
            lbl = processedHands.multi_handedness[idx].classification[0].label
            handRow = list(np.array([[lm.x, lm.y, lm.z] for lm in handLms.landmark]).flatten())

            # Add image definition and handedness to beginning of array
            handRow.insert(0, lbl + "." + imageDefinition)
            #print(handRow)

            # Export to CSV
            with open(datasetFile, mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(handRow)

    # Render each frame
    cv2.imshow("Data Collection", frame)