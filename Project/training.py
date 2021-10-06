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

datasetFile = "./datasets/thumbs.csv"
modelFile = "./models/thumbs_gb.pkl"

# Initialize dataset
lmf = pd.read_csv(datasetFile)
#print(lmf)
#print(lmf[lmf["definition"] == "Right.thumbs_up"])

#n = 0 # index to obtain
#imageDefinition = lmf.iloc[n, 0]
#landmarks = lmf.iloc[n, 1:]
#landmarks = np.asarray(landmarks)
#landmarks = landmarks.astype("float").reshape(-1, 3)

landmarks = lmf.drop("definition", axis=1)
definitions = lmf["definition"]

#print("Image definition: {}".format(definitions))
#print("Landmarks shape: {}".format(landmarks.shape))
#print("First 4 Landmarks: {}".format(landmarks[:4]))

lm_train, lm_test, def_train, def_test = train_test_split(landmarks, definitions, test_size=0.3, random_state=1337)

# Initialize pipelines
pipelines = {
    "gb":make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    "rf":make_pipeline(StandardScaler(), RandomForestClassifier()),
    "lr":make_pipeline(StandardScaler(), LogisticRegression()),
    "rc":make_pipeline(StandardScaler(), RidgeClassifier())
}

# Train pipelines into models
models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(lm_train, def_train)
    models[algo] = model

#print(models["gc"].predict(lm_test))

# Evaluate accuracy of models
print("[*] Displaying accuracies for each algorithm:")
for algo, model in models.items():
    yhat = model.predict(lm_test)
    print(algo, accuracy_score(def_test, yhat))

# Serialize best model (choosing 'lr')
with open(modelFile, 'wb') as f:
    pickle.dump(models['gb'], f)

print("[*] Model trained and saved to {}".format(modelFile))
print(lm_train)
print(def_train)