import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer, Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

# Define Custom Layers
class TransposeLayer(Layer):
    def __init__(self, perm, **kwargs):
        super(TransposeLayer, self).__init__(**kwargs)
        self.perm = perm

    def call(self, inputs):
        return tf.transpose(inputs, perm=self.perm)

    def get_config(self):
        config = super(TransposeLayer, self).get_config()
        config.update({"perm": self.perm})
        return config

class ReshapeLayer(Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, self.target_shape)

    def get_config(self):
        config = super(ReshapeLayer, self).get_config()
        config.update({"target_shape": self.target_shape})
        return config

class CustomMessagePassing(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(CustomMessagePassing, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.self_dense = Dense(units)
        self.neighbor_dense = Dense(units)

    def call(self, inputs):
        features, adjacency = inputs
        self_features = self.self_dense(features)
        neighbor_features = tf.matmul(adjacency, self.neighbor_dense(features))
        combined_features = self_features + neighbor_features
        if self.activation:
            combined_features = self.activation(combined_features)
        return combined_features

    def get_config(self):
        config = super(CustomMessagePassing, self).get_config()
        config.update({
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation),
        })
        return config
    
def predict_next_day(new_day_data, historical_data_path='psa_hack_2024/data/waiting_times_updated.csv', 
                   model_path='psa_hack_2024/model/best_model_custom_gnn.keras', scaler_path='psa_hack_2024/model/scaler.pkl',
                   adjacency_matrix_path='psa_hack_2024/data/port_relationships.csv', time_steps=30, num_ports=50):
    """
    Predict the next day's waiting times for each port based on new data and update historical data.
    
    Parameters:
    - new_day_data: list or array of shape (num_ports,)
        The waiting times for the new day for each port.
    - historical_data_path: str
        Path to the CSV file containing historical data of waiting times.
    - model_path: str
        Path to the trained Keras model file.
    - scaler_path: str
        Path to the fitted scaler file.
    - adjacency_matrix_path: str
        Path to the CSV file containing the adjacency matrix of port relationships.
    - time_steps: int
        The number of days to look back for predicting the next day.
    - num_ports: int
        The number of ports being considered.

    Returns:
    - predicted_waiting_times: ndarray of shape (num_ports,)
        The predicted waiting times for the next day for each port.
    """
    
    # Step 1: Load the trained model and scaler
    model = load_model(model_path, custom_objects={
        'TransposeLayer': TransposeLayer,
        'ReshapeLayer': ReshapeLayer,
        'CustomMessagePassing': CustomMessagePassing
    })
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Step 2: Load historical data and update it with new day data
    if os.path.exists(historical_data_path):
        historical_df = pd.read_csv(historical_data_path, header=None)
    else:
        raise FileNotFoundError(f"Historical data file not found at {historical_data_path}.")
    
    # Convert historical data to a numpy array
    historical_data = historical_df.values
    
    # Ensure the new_day_data is a numpy array of the correct shape
    new_day_data = np.array(new_day_data).reshape(1, -1)
    
    if new_day_data.shape[1] != num_ports:
        raise ValueError(f"Expected new day data to have {num_ports} values, but got {new_day_data.shape[1]}")
    
    # Append the new day's data to the historical data
    updated_historical_data = np.vstack([historical_data, new_day_data])
    
    # Keep only the most recent time_steps days
    if updated_historical_data.shape[0] > time_steps:
        updated_historical_data = updated_historical_data[-time_steps:]
    
    # Step 3: Scale the updated historical data
    updated_historical_data_scaled = scaler.transform(updated_historical_data)
    
    # Reshape to the required input format for the model: (1, time_steps, num_ports)
    X_new = np.expand_dims(updated_historical_data_scaled, axis=0)
    
    # Step 4: Load and normalize the adjacency matrix
    adjacency_matrix = pd.read_csv(adjacency_matrix_path, header=None).values
    epsilon = 1e-8
    row_sums = adjacency_matrix.sum(axis=1, keepdims=True) + epsilon
    adjacency_matrix_normalized = adjacency_matrix / row_sums
    
    # Repeat the adjacency matrix for a single input: shape (1, num_ports, num_ports)
    adj_new = np.expand_dims(adjacency_matrix_normalized, axis=0)
    
    # Step 5: Make the prediction using the trained model
    prediction_scaled = model.predict([X_new, adj_new])  # Shape: (1, num_ports)
    
    # Step 6: Inverse transform the predictions
    predicted_waiting_times = scaler.inverse_transform(prediction_scaled)  # Shape: (1, num_ports)
    
    # Step 7: Save the updated historical data back to the CSV
    updated_historical_df = pd.DataFrame(updated_historical_data)
    updated_historical_df.to_csv(historical_data_path, index=False, header=False)
    
    # Return the predicted waiting times for the next day
    return predicted_waiting_times.flatten()
