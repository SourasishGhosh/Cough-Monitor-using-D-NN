from keras_tuner import RandomSearch
import numpy as np
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load dataset (use your feature extraction pipeline)
X = np.load(r"D:\PERSONAL PROJECTS\Cough Monitor\my-docker-project\features.npy",allow_pickle=True)   # or from data_pipeline.load_dataset()
y = np.load(r"D:\PERSONAL PROJECTS\Cough Monitor\my-docker-project\labels.npy",allow_pickle=True)

# Load encoder to know num_classes
encoder = joblib.load("saved_models/label_encoder.pkl")
num_classes = len(encoder.classes_)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)


# --- Define model-building function ---
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        hp.Int('units1', min_value=64, max_value=256, step=64),
        activation='relu',
        input_shape=(X.shape[1],)  # (40,)
    ))
    model.add(keras.layers.Dropout(hp.Float('dropout1', 0.2, 0.5, step=0.1)))
    model.add(keras.layers.Dense(
        hp.Int('units2', min_value=32, max_value=256, step=32),
        activation='relu'
    ))
    model.add(keras.layers.Dropout(hp.Float('dropout2', 0.2, 0.5, step=0.1)))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer="adam",
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Define tuner ---
tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,              # number of different models to try
    executions_per_trial=2,    # repeat each trial for stability
    directory="tuner_results",
    project_name="coswara_optimization"
)

# --- Run search ---
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# --- Get best model ---
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate
loss, acc = best_model.evaluate(X_val, y_val, verbose=0)
print(f"Best Validation Accuracy: {acc:.4f}")

# Save best model
best_model.save("coswara_dnn_tuned.h5")
