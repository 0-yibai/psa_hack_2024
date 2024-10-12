import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer, Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
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

# Parameters
num_ports = 50
time_steps = 30
num_days = 1000
num_samples_seq = num_days - time_steps  # 970
lstm_units = 64
gnn_units = 64

# Ensure the model directory exists
os.makedirs('../model/', exist_ok=True)

# Load Waiting Times from CSV
waiting_times_df = pd.read_csv('data/waiting_times.csv', header=None)

# Convert to NumPy Array
data = waiting_times_df.values

# Scale Waiting Times
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Save the fitted scaler to 'scaler.pkl'
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Create Input Sequences and Targets
X = np.zeros((num_samples_seq, time_steps, num_ports)) 
y = np.zeros((num_samples_seq, num_ports))
for i in range(num_samples_seq):
    X[i] = data_scaled[i:i + time_steps]  # Past 30 days
    y[i] = data_scaled[i + time_steps]    # Next day

# Load Adjacency Matrix from CSV
adjacency_df = pd.read_csv('data/port_relationships.csv', header=None)

# Convert to NumPy Array
adjacency_matrix = adjacency_df.values

# Normalize Adjacency Matrix Row-Wise
epsilon = 1e-8
row_sums = adjacency_matrix.sum(axis=1, keepdims=True) + epsilon
adjacency_matrix_normalized = adjacency_matrix / row_sums

# Repeat Adjacency Matrix for All Samples
adjacency_batch = np.repeat(adjacency_matrix_normalized[np.newaxis, :, :], num_samples_seq, axis=0)

# Split Data into Training, Validation, and Test Sets
train_size = int(0.7 * num_samples_seq) 
val_size = int(0.15 * num_samples_seq)  
test_size = num_samples_seq - train_size - val_size

X_train = X[:train_size]
y_train = y[:train_size]
adj_train = adjacency_batch[:train_size]

X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
adj_val = adjacency_batch[train_size:train_size + val_size]

X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]
adj_test = adjacency_batch[train_size + val_size:]

# Define the Model Architecture
waiting_time_input = Input(shape=(time_steps, num_ports), name='waiting_time_input') 
adjacency_input = Input(shape=(num_ports, num_ports), name='adjacency_input')  

# Process Waiting Times with LSTM
waiting_time_transposed = TransposeLayer(perm=[0, 2, 1])(waiting_time_input)

# Reshape to (batch_size * 50, 30, 1)
reshape_layer = ReshapeLayer(target_shape=(-1, time_steps, 1))(waiting_time_transposed)

# LSTM Layer to Capture Temporal Features
lstm_out = LSTM(units=lstm_units, return_sequences=False, name='lstm_port')(reshape_layer)

# Reshape Back to (batch_size, 50, 64)
port_features = ReshapeLayer(target_shape=(-1, num_ports, lstm_units))(lstm_out)

# Custom Message Passing Layer for Spatial Processing
gnn_layer_1 = CustomMessagePassing(units=gnn_units, activation='relu')([port_features, adjacency_input])
gnn_layer_1 = Dropout(0.2)(gnn_layer_1)

# Add Another Message Passing Layer
gnn_layer_2 = CustomMessagePassing(units=gnn_units, activation='relu')([gnn_layer_1, adjacency_input])
gnn_layer_2 = Dropout(0.2)(gnn_layer_2)

# Flatten the GNN Output for Prediction
flattened_gnn = ReshapeLayer(target_shape=(-1, num_ports * gnn_units))(gnn_layer_2)

# Dense Layers for Final Prediction
dense_1 = Dense(units=256, activation='relu')(flattened_gnn)
dense_1 = Dropout(0.3)(dense_1)
output = Dense(units=num_ports, activation='linear', name='output_layer')(dense_1)

# Define the Model
model = Model(inputs=[waiting_time_input, adjacency_input], outputs=output)

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Print Model Summary
model.summary()

# Define Callbacks for Early Stopping and Model Checkpointing
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('model/best_model_custom_gnn.keras', monitor='val_loss', save_best_only=True)

# Fit the Model
history = model.fit(
    x=[X_train, adj_train],
    y=y_train,
    batch_size=32,
    epochs=100,  # Early stopping will handle actual stopping
    validation_data=([X_val, adj_val], y_val),
    callbacks=[early_stop, checkpoint]
)

# Evaluate the Model on Test Set
test_loss = model.evaluate(
    x=[X_test, adj_test],
    y=y_test
)
print(f"Test Loss (MSE): {test_loss}")

# Make Predictions on Test Set
predictions = model.predict([X_test, adj_test])

# Inverse transform y_test and predictions
y_test_original = scaler.inverse_transform(y_test)        
predictions_original = scaler.inverse_transform(predictions) 

# Calculate Additional Metrics
mae = mean_absolute_error(y_test_original, predictions_original)
mse = mean_squared_error(y_test_original, predictions_original)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot Training and Validation Loss
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# # Example of Making a Prediction for a Single Sample
# single_waiting_times = X_test[0].reshape(1, time_steps, num_ports)  # (1, 30, 50)
# single_adjacency = adj_test[0].reshape(1, num_ports, num_ports)      # (1, 50, 50)

# # Make the Prediction
# predicted_waiting_times_single = model.predict([single_waiting_times, single_adjacency])

# # Inverse Scale the Prediction
# predicted_waiting_times_single_original = scaler.inverse_transform(predicted_waiting_times_single)

# # Handle Negative Predictions
# predicted_waiting_times_single_original = np.maximum(predicted_waiting_times_single_original, 0)

# print("Predicted Waiting Times for the Next Day (Single Sample):")
# print(predicted_waiting_times_single_original)

# # Visualization for a Specific Port
# port_index = 13  # 0-based indexing

# plt.figure(figsize=(6,6))
# plt.scatter(y_test_original[:, port_index], predictions_original[:, port_index], alpha=0.5)
# plt.plot([y_test_original.min(), y_test_original.max()],
#          [y_test_original.min(), y_test_original.max()],
#          'r--')  # Diagonal line
# plt.xlabel('Actual Waiting Time')
# plt.ylabel('Predicted Waiting Time')
# plt.title(f'Actual vs. Predicted Waiting Times for Port {port_index + 1}')
# plt.show()
