# lstm_model.py

import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
import joblib 

TIMESTEPS = 10 # How many events to look at in one sequence

def extract_features(events_json):
    """Extracts features (dwell and flight) from raw event data."""
    if len(events_json) < 20: return None
    df = pd.DataFrame(events_json)
    
    # Dwell Time
    downs = df[df['action'] == 'keydown'].copy()
    ups = df[df['action'] == 'keyup'].copy()
    merged = pd.merge(downs, ups, on='key', suffixes=('_down', '_up'))
    merged = merged[merged['timestamp_up'] > merged['timestamp_down']]
    merged['dwell_time'] = merged['timestamp_up'] - merged['timestamp_down']
    merged = merged[merged['dwell_time'] < 1000]
    
    # We will just use dwell times as the primary feature sequence for simplicity
    dwell_times = merged['dwell_time'].values
    if len(dwell_times) < TIMESTEPS + 1: return None
    
    return dwell_times.reshape(-1, 1) # Return as a 2D array [samples, features]

def create_sequences(data):
    """Converts a list of features into 3D sequences for an LSTM."""
    X = []
    for i in range(len(data) - TIMESTEPS):
        X.append(data[i:i + TIMESTEPS])
    return np.array(X)

def build_model(input_shape):
    """Builds the LSTM Autoencoder model."""
    model = keras.Sequential([
        # Encoder
        layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        layers.LSTM(32, activation='relu', return_sequences=False),
        layers.RepeatVector(input_shape[0]),
        # Decoder
        layers.LSTM(32, activation='relu', return_sequences=True),
        layers.LSTM(64, activation='relu', return_sequences=True),
        layers.TimeDistributed(layers.Dense(input_shape[1]))
    ])
    model.compile(optimizer='adam', loss='mae') # Mean Absolute Error
    return model

def train_user_model(user_id, raw_events):
    """Main function to train and save a model for a user."""
    
    # 1. Extract features
    features = extract_features(raw_events)
    if features is None:
        print("Feature extraction failed, insufficient data.")
        return None, None

    # 2. Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    
    # 3. Create 3D sequences
    sequences = create_sequences(scaled_data)
    if len(sequences) == 0:
        print("Sequence creation failed, insufficient data.")
        return None, None
        
    n_features = sequences.shape[2] # Should be 1
    
    # 4. Build and train the model
    model = build_model(input_shape=(TIMESTEPS, n_features))
    print(f"Starting model training for user {user_id}...")
    model.fit(sequences, sequences, epochs=50, batch_size=32, verbose=1)
    print("Model training complete.")
    
    # 5. Save the model and the scaler
    model_path = f"instance/user_{user_id}_model.keras" # Use the modern .keras format
    scaler_path = f"instance/user_{user_id}_scaler.joblib"
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    return model_path, scaler_path

def get_model_score(user_profile, raw_events):
    """Loads a user's model and scores their live data."""
    try:
        model = keras.models.load_model(user_profile.model_path)
        scaler = joblib.load(user_profile.scaler_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0.0 

    features = extract_features(raw_events)
    if features is None: return 0.1 

    scaled_data = scaler.transform(features)
    sequences = create_sequences(scaled_data)
    if len(sequences) == 0: return 0.1 

    reconstruction_error = model.evaluate(sequences, sequences, verbose=0)
    
    # Convert error to a trust score
    trust_score = max(0, 1 - (reconstruction_error * 2)) 
    
    print(f"Reconstruction Error: {reconstruction_error:.4f}, Trust Score: {trust_score:.2f}")
    return round(trust_score, 2)