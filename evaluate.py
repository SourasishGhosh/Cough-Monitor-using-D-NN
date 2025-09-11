import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os 

# Load artifacts
model = load_model("saved_models/coswara_dnn.h5")
encoder = joblib.load("saved_models/label_encoder.pkl")

X_test = np.load("saved_models/X_test.npy")
y_test = np.load("saved_models/y_test.npy")

# Predict
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Reports
print(classification_report(y_true, y_pred_classes, target_names=encoder.classes_))
print(confusion_matrix(y_true, y_pred_classes))
