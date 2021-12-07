import pandas as pd
# ML pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Four training algorithms
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
# Serialization
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt

datasetFile = "./datasets/rhand_alphabet_minhd2_co.csv"
modelFile = "./models/rhand_alphabet_minhd2"

# Initialize dataset
lmf = pd.read_csv(datasetFile)

landmarks = lmf.drop("definition", axis=1)
definitions = lmf["definition"]

row = 223

for i in range(0, 63, 3):
    plt.plot(landmarks.iloc[0:221, i], landmarks.iloc[0:221, i+1], 'b+')
    plt.plot(landmarks.iloc[223:410, i], landmarks.iloc[223:410, i+1], 'r+')
    plt.xlabel(definitions[row])
    plt.ylabel(definitions[row])

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.gca().invert_yaxis()
plt.show()