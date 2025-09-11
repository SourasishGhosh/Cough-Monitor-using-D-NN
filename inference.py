import librosa
import numpy as np
from tensorflow import keras
import joblib
import os

# --- Build the same architecture you trained ---
def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(40,)),             # Input layer
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(7, activation="softmax")  # adjust '7' to num_classes
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# Load trained artifacts
model = build_model()
model.load_weights("saved_models/coswara_dnn.h5")
encoder = joblib.load("saved_models/label_encoder.pkl")

#example_path = r"D:\PERSONAL PROJECTS\Cough Monitor\my-docker-project\Test Sample\my_cough_070925.wav"
example_path = "/app/test_samples/my_cough_070925.wav"


def predict_from_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # Take mean across time to match training
    mfcc_mean = np.mean(mfcc, axis=1)   # shape (40,)

    # Reshape for model
    X = mfcc_mean[np.newaxis, :]        # shape (1, 40)

    # Predict
    pred = model.predict(X)
    pred_class = encoder.inverse_transform([pred.argmax(axis=1)[0]])
    return pred_class[0]


# Example
#print(predict_from_audio(example_path))
