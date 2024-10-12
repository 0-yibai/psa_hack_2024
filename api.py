# api.py
from flask import Flask, request, jsonify
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
from flask_cors import CORS

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

app = Flask(__name__)
CORS(app)

# Load the model and scaler when the app starts
model = load_model(
    'model/best_model_custom_gnn.keras',
    custom_objects={
        'TransposeLayer': TransposeLayer,
        'ReshapeLayer': ReshapeLayer,
        'CustomMessagePassing': CustomMessagePassing
    }
)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load and normalize the adjacency matrix
adjacency_matrix = pd.read_csv('data/port_relationships.csv', header=None).values
epsilon = 1e-8
row_sums = adjacency_matrix.sum(axis=1, keepdims=True) + epsilon
adjacency_matrix_normalized = adjacency_matrix / row_sums
adj_new = np.expand_dims(adjacency_matrix_normalized, axis=0)  # Shape: (1, num_ports, num_ports)

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract new day's data
        new_day_data = data.get('new_day_data')

        if not new_day_data or len(new_day_data) != 50:
            return jsonify({'error': 'Please provide a list of 50 waiting times.'}), 400

        # Convert to numpy array
        new_day_data = np.array(new_day_data).reshape(1, -1)

        # Load historical data
        historical_data_path = 'data/waiting_times_updated.csv'
        if os.path.exists(historical_data_path):
            historical_df = pd.read_csv(historical_data_path, header=None)
        else:
            historical_df = pd.read_csv('data/waiting_times.csv', header=None)
        historical_data = historical_df.values

        # Append new data and keep last 30 days
        updated_historical_data = np.vstack([historical_data, new_day_data])
        if updated_historical_data.shape[0] > 30:
            updated_historical_data = updated_historical_data[-30:]

        # Scale the data
        updated_historical_data_scaled = scaler.transform(updated_historical_data)

        # Prepare input for the model
        X_new = np.expand_dims(updated_historical_data_scaled, axis=0)

        # Make prediction
        prediction_scaled = model.predict([X_new, adj_new])  # Shape: (1, num_ports)
        predicted_waiting_times = scaler.inverse_transform(prediction_scaled).flatten()

        # Update historical data file
        updated_historical_df = pd.DataFrame(updated_historical_data)
        updated_historical_df.to_csv(historical_data_path, index=False, header=False)

        # Return the predictions as JSON
        return jsonify({'predicted_waiting_times': predicted_waiting_times.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
