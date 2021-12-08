import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import tkinter as tk
import time

from PIL import Image, ImageTk
from pynput import keyboard
from pynput.keyboard import Key, Controller
from tkinter import ttk

import warnings
warnings.filterwarnings("ignore")

modelFile = "./models/lbfr_minhd2_lr.pkl"

# Intialize vision
width, height = 800, 600
cap = cv2.VideoCapture(0)
time.sleep(1.000)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Initialize GUI
root = tk.Tk()
root.attributes('-topmost', True)
root.title("Sign Sign Revolution")
root.bind("<Escape>", lambda e: root.quit())
WEBCAM_PANE = tk.Label(root)
WEBCAM_PANE.pack(side="left")
SETTINGS_PANE = ttk.Notebook(root)
SETTINGS_PANE.pack(side="right")

# Intialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Initalize model
with open(modelFile, 'rb') as f:
    global currentModel
    currentModel = pickle.load(f)

def on_press(key):
    if (key == Key.space):
        if (displayDefinition == "Right.letter_f"):
            print("Forward")
            outputKeyboard.press(Key.up)
            outputKeyboard.release(Key.up)
        elif (displayDefinition == "Right.letter_b"):
            print("Backward")
            outputKeyboard.press(Key.down)
            outputKeyboard.release(Key.down)
        elif (displayDefinition == "Right.letter_l"):
            print("Left")
            outputKeyboard.press(Key.left)
            outputKeyboard.release(Key.left)
        elif (displayDefinition == "Right.letter_r"):
            print("Right")
            outputKeyboard.press(Key.right)
            outputKeyboard.release(Key.right)
        else:
            print(displayDefinition)

# Initialize keyboard
outputKeyboard = Controller()
listener = keyboard.Listener(
    on_press=on_press)
listener.start()

def frameLoop():
    global displayDefinition

    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    processedHands = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Frame information
    h, w, c = frame.shape
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h

    if processedHands.multi_hand_landmarks:
        for idx, handLms in enumerate(processedHands.multi_hand_landmarks):
            lbl = processedHands.multi_handedness[idx].classification[0].label
            handRow = list(np.array([[lm.x, lm.y, lm.z] for lm in handLms.landmark]).flatten())

            X = pd.DataFrame([handRow])
            displayDefinition = currentModel.predict(X)[0]

    # Draw hand landmarks on each frame
    if processedHands.multi_hand_landmarks:
        for idx, handLms in enumerate(processedHands.multi_hand_landmarks):
            lbl = processedHands.multi_handedness[idx].classification[0].label
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
            #cv2.putText(frame, text = lbl, org = (x_min, y_min), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 0), thickness = 2, lineType = cv2.LINE_AA)
            cv2.putText(frame, text = displayDefinition, org = (0, 100), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 0), thickness = 2, lineType = cv2.LINE_AA)
            mpDraw.draw_landmarks(frame, handLms)

    # Render each frame on GUI
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    WEBCAM_PANE.imgtk = imgtk
    WEBCAM_PANE.configure(image=imgtk)
    WEBCAM_PANE.after(10, frameLoop)

def main():
    frameLoop()
    root.mainloop()

if __name__ == "__main__":
    main()