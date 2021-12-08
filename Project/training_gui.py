import csv
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import tkinter as tk
import time
import os

from cv2 import data
from PIL import Image, ImageTk
from tkinter import simpledialog
from tkinter import ttk
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Intialize vision
width, height = 800, 600
cap = cv2.VideoCapture(0)
time.sleep(1.000)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Initialize GUI
root = tk.Tk()
root.title("Training GUI")
root.bind("<Escape>", lambda e: root.quit())
WEBCAM_PANE = tk.Label(root)
WEBCAM_PANE.pack(side="left")
SETTINGS_PANE = ttk.Notebook(root)
SETTINGS_PANE.pack(side="right")

# Initialize variables and strings
datasetName = tk.StringVar(value="rhand_alphabet")
netID = tk.StringVar(value="netid0")
datasetDefinition = tk.StringVar(value="letter_a")
algorithm = tk.StringVar(value="ALGORITHM")
datasetFile = tk.StringVar(value="./datasets/" + datasetName.get() + "_" + netID.get() + ".csv")
modelFile = tk.StringVar(value="./models/" + datasetName.get() + "_" + netID.get() + "_" + algorithm.get() + ".pkl")
overwriteData = tk.BooleanVar(value=False)
recordButtonMessage = tk.StringVar(value="Start recording")
recording = tk.BooleanVar(value=False)
identifyButtonMessage = tk.StringVar(value="Start identify")
identifying = tk.BooleanVar(value=False)
#customModelFile = tk.BooleanVar(value=False)

# Initialize data collection GUI
DATA_COLLECTION_TAB = tk.Frame(SETTINGS_PANE)
tk.Label(DATA_COLLECTION_TAB, text="Dataset name: ").grid(row=0, column=0, sticky='w')
tk.Entry(DATA_COLLECTION_TAB, textvariable=datasetName).grid(row=0, column=1, sticky='w')
tk.Label(DATA_COLLECTION_TAB, text="NetID or unique ID: ").grid(row=1, column=0, sticky='w')
tk.Entry(DATA_COLLECTION_TAB, textvariable=netID).grid(row=1, column=1, sticky='w')
tk.Label(DATA_COLLECTION_TAB, textvariable=datasetFile).grid(row=2, column=0, sticky='w')
tk.Button(DATA_COLLECTION_TAB, text="Update", command=lambda: setDatasetFile()).grid(row=2, column=1, sticky='w')
tk.Label(DATA_COLLECTION_TAB, text="Definition (value to record): ").grid(row=4, column=0, sticky='w')
tk.Entry(DATA_COLLECTION_TAB, textvariable=datasetDefinition).grid(row=4, column=1, sticky='w')
tk.Button(DATA_COLLECTION_TAB, textvariable=recordButtonMessage, command=lambda: toggleRecord()).grid(row=5, column=0, sticky='w')
# TODO: Add live recording status
# TODO: Add dropdown to save only points for left hand, right hand, or both hands

# Intialize model training GUI
MODEL_TRAINING_TAB = tk.Frame(SETTINGS_PANE)
tk.Label(MODEL_TRAINING_TAB, text="Use dataset file: ").grid(row=0, column=0, sticky='w')
tk.Entry(MODEL_TRAINING_TAB, textvariable=datasetFile).grid(row=0, column=1, sticky='w')
#tk.Checkbutton(MODEL_TRAINING_TAB, text="Set custom model file", variable=customModelFile, command=lambda: toggleCustomModelFile()).grid(row=1, sticky='w')
#tk.Label(MODEL_TRAINING_TAB, text="Save to model file: ").grid(row=2, column=0, sticky='w')
#tk_entry_modelFile = tk.Entry(MODEL_TRAINING_TAB, textvariable=modelFile, state="disabled")
#tk_entry_modelFile.grid(row=2, column=1, sticky='w')
tk.Button(MODEL_TRAINING_TAB, text="Train models", command=lambda: trainModels()).grid(row=1, column=0, sticky='w')

IDENTIFY_TAB = tk.Frame(SETTINGS_PANE)
tk.Label(IDENTIFY_TAB, text="Use model file: ").grid(row=0, column=0, sticky='w')
tk.Entry(IDENTIFY_TAB, textvariable=modelFile).grid(row=0, column=1, sticky='w')
tk.Button(IDENTIFY_TAB, textvariable=identifyButtonMessage, command=lambda: toggleIdentify()).grid(row=1, column=0, sticky='w')

# Initialize all tabs
SETTINGS_PANE.add(DATA_COLLECTION_TAB, text="Data Collection")
SETTINGS_PANE.add(MODEL_TRAINING_TAB, text="Model Training")
SETTINGS_PANE.add(IDENTIFY_TAB, text="Identify")

# Intialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

def toggleIdentify():
    if not os.path.exists(modelFile.get()):
        tk.messagebox.showwarning(title="Warning", message="Please enter valid model file location")
        return
    identifying.set(not identifying.get())
    if (identifying.get()):
        identifyButtonMessage.set("Stop identify")
        print("[*] Started identifying")
        # Initialize model
        with open(modelFile.get(), 'rb') as f:
            global currentModel
            currentModel = pickle.load(f)
    else:
        identifyButtonMessage.set("Start identify")
        print("[*] Stopped identifying")


def toggleRecord():
    if (datasetFile.get() == "./datasets/rhand_alphabet_netid0.csv"):
        tk.messagebox.showwarning(title="Warning", message="Please set your ID")
        return
    recording.set(not recording.get())
    if (recording.get()):
        recordButtonMessage.set("Stop recording")
        print("[*] Started recording")
    else:
        recordButtonMessage.set("Start recording")
        print("[*] Stopped recording")

#def toggleCustomModelFile():
#    if (customModelFile.get()):
#        tk_entry_modelFile.configure(state="normal")
#    else:
#        tk_entry_modelFile.configure(state="disabled")

def setDatasetFile():
    datasetFile.set("./datasets/" + datasetName.get() + "_" + netID.get() + ".csv")

def initializeSave():
    # Initialize CSV columns and file if necessary
    headers = ['definition']
    for idx in range(0, 21): # 21 hand landmark coordinates
        headers += ['x{}'.format(idx),'y{}'.format(idx),'z{}'.format(idx)]
    if not os.path.exists(datasetFile.get()):
        print("[*] Initializing new dataset in file {}".format(datasetFile.get()))
        with open(datasetFile.get(), mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(headers)

def trainModels():
    lmf = pd.read_csv(datasetFile.get())
    landmarks = lmf.drop("definition", axis=1)
    definitions = lmf["definition"]
    lm_train, lm_test, def_train, def_test = train_test_split(landmarks, definitions, test_size=0.3, random_state=1337)

    # Initialize pipelines with algorithms
    pipelines = {
        "gb":make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        "rf":make_pipeline(StandardScaler(), RandomForestClassifier()),
        "lr":make_pipeline(StandardScaler(), LogisticRegression()),
        "rc":make_pipeline(StandardScaler(), RidgeClassifier())
    }

    # Train algorithms into models
    models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(lm_train.values, def_train)
        models[algo] = model

    # Evaluate accuracy of models
    print("[*] Displaying accuracies for each algorithm:")
    SUMMARY_WINDOW = tk.Toplevel(root)
    SUMMARY_WINDOW.title("Model Results")
    i = 0
    for algo, model in models.items():
        yhat = model.predict(lm_test)
        score = accuracy_score(def_test, yhat)
        print(algo, score)
        tk.Label(SUMMARY_WINDOW, text=algo).grid(row=i, column=0, sticky='w')
        tk.Label(SUMMARY_WINDOW, text=score).grid(row=i, column=1, sticky='w')
        tk.Button(SUMMARY_WINDOW, text="Select", command=lambda: algorithm.set(algo)).grid(row=i, column=2, sticky='w')
        i += 1
    root.wait_variable(algorithm)
    SUMMARY_WINDOW.destroy()

    # Serialize selected model
    modelFile.set(simpledialog.askstring(
        title="Save Model File",
        prompt="Save model file: ",
        initialvalue="./models/" + datasetName.get() + "_" + netID.get() + "_" + algorithm.get() + ".pkl"
    ))
    try:
        if os.path.exists(modelFile.get()):
            print("[!] Warning: Model file {} exists and will be overwritten".format(modelFile.get()))
            if not tk.messagebox.askokcancel(
                title="Warning",
                icon="warning",
                message="Model file {} exists and will be overwritten".format(modelFile.get())
            ):
                return
        with open(modelFile.get(), 'wb') as f:
            pickle.dump(models[algorithm.get()], f)
    except:
        print("[!] Error: Failed to save model")

    print("[*] Model trained and saved to {}".format(modelFile.get()))

def frameLoop():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    processedHands = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Frame information
    h, w, c = frame.shape
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h

    # Record and collect all data points into a CSV dataset file
    if processedHands.multi_hand_landmarks and (recording.get() or identifying.get()):
        for idx, handLms in enumerate(processedHands.multi_hand_landmarks):
            lbl = processedHands.multi_handedness[idx].classification[0].label
            handRow = list(np.array([[lm.x, lm.y, lm.z] for lm in handLms.landmark]).flatten())

            if recording.get():
                # Export to CSV
                handRow.insert(0, lbl + "." + datasetDefinition.get())
                initializeSave()
                with open(datasetFile.get(), mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(handRow)
            elif identifying.get():
                #handRow = np.array(handRow)
                # Predict model
                #handRowFrame = pd.DataFrame(handRow.reshape(-1, len(handRow)), columns = ['x0','y0','z0','x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4','x5','y5','z5','x6','y6','z6','x7','y7','z7','x8','y8','z8','x9','y9','z9','x10','y10','z10','x11','y11','z11','x12','y12','z12','x13','y13','z13','x14','y14','z14','x15','y15','z15','x16','y16','z16','x17','y17','z17','x18','y18','z18','x19','y19','z19','x20','y20','z20'])
                #print(handRowFrame)
                #handRowFrame = StandardScaler().fit_transform(handRowFrame)
                #print(handRowFrame)
                #X = np.transpose(StandardScaler().fit_transform(np.array(handRow).reshape(-1, 1)))[0]
                #print(X)
                X = pd.DataFrame([handRow])
                #print(X)
                displayDefinition = currentModel.predict(X)[0]
                #probability = currentModel.predict_proba(X)[0]

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
            #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, text=lbl, org=(x_min, y_min), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            if identifying.get():
                cv2.putText(frame, text=displayDefinition, org=(x_max, y_max), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
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

# "I hate Tkinter" - WhiteHoodHacker