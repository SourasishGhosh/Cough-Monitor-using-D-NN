import os
import json
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers, utils


print("--- Starting Data Processing and Feature Extraction ---")

# Define base directory and sample rate
BASE_DIR = r"D:\PERSONAL PROJECTS\ML Project Ideas\Multimodal Cough-Based Respiratory Health Predictor\Kaggle_Coswara\coswara_wav"
SAMPLE_RATE = 16000

def load_audio(file_path, sr=SAMPLE_RATE):
    """Load and normalize audio."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        if y.size == 0:
            return None, None
        y = librosa.util.normalize(y)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def extract_features(y, sr, n_mfcc=40):
    """Extract MFCC features from audio."""
    if y is None or sr is None:
        return None
    if len(y) < sr * 0.2:  # Ensure audio is at least 0.2 seconds long
        return None
    
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfccs.shape[1] == 0:
            return None
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

data = []

# Loop through each batch folder (e.g., 20200413)
for batch_folder in os.listdir(BASE_DIR):
    batch_path = os.path.join(BASE_DIR, batch_folder)
    if not os.path.isdir(batch_path):
        continue
    
    # Loop through participant folders
    for participant_id in os.listdir(batch_path):
        participant_path = os.path.join(batch_path, participant_id)
        if not os.path.isdir(participant_path):
            continue
        
        meta_path = os.path.join(participant_path, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        label = metadata.get("covid_status", "unknown")
        
        # Loop through audio files for the participant
        for file in os.listdir(participant_path):
            if file.endswith(".wav"):
                file_path = os.path.join(participant_path, file)
                
                y, sr = load_audio(file_path)
                features = extract_features(y, sr)
                
                if features is None:
                    print(f"Skipping {file_path} (empty, too short, or error)")
                    continue
                
                data.append([batch_folder, participant_id, file, features, label])

# Create DataFrame
df = pd.DataFrame(data, columns=["batch_folder", "participant_id", "file", "features", "label"])
print(df.head())

# Saving features and labels for model training
X = np.array(df['features'].tolist())
y = df['label']

print(f"\nFeature array shape: {X.shape}")
print(f"Labels array shape: {y.shape}")
print(f"Unique labels: {np.unique(y)}")

# --- Model Training and Evaluation ---
print("\n--- Starting Model Training ---")

# Encode labels for training
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = utils.to_categorical(y_encoded)


# for a DNN must be 2D: (samples, features).

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# ----------------------------
# Dense NN Model Architecture
# ----------------------------
model = models.Sequential([
    # Input shape is (40,) for 40 MFCC features
    layers.Input(shape=(40,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    # Output layer with a node for each class
    layers.Dense(y_cat.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------------
# Train the model
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32
)

# ----------------------------
# Evaluate the model
# ----------------------------
print("\n--- Evaluating Model ---")
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# --- Saving the Model and Assets ---
print("\n--- Saving Model and Assets ---")

# Create a directory to save the model and other assets
os.makedirs("saved_models", exist_ok=True)

# Save the trained model
model.save("saved_models/coswara_dnn.keras")

# Save the label encoder
joblib.dump(encoder, "saved_models/label_encoder.pkl")

# Save the test data for later evaluation (e.g., for making predictions)
np.save("saved_models/X_test.npy", X_test)
np.save("saved_models/y_test.npy", y_test)

print("Model, encoder, and test data saved successfully.")
