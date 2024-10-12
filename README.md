
# PSA Port Waiting Time Prediction

This project is a web-based application for predicting port waiting times, built using React for the front-end and a machine learning model to handle predictions. The web app allows users to input current waiting times for different ports and predict the next day's waiting time.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installing Dependencies](#installing-dependencies)
  - [Running the Application](#running-the-application)
- [Model Training](#model-training)
- [Technologies Used](#technologies-used)
- [License](#license)

## Project Structure

```
psa_hack_2024/
├── data/
│   ├── port_relationship.csv
│   ├── waiting_time_updated.csv
│   └── waiting_time.csv
├── model/
│   ├── best_model_custom_gnn.keras
│   └── scaler.pkl
├── prediction_app/
│   ├── public/
│   └── src/
├── scripts/
│   ├── predict_func.py
│   └── train_model.py
├── simulation/
│   ├── crisis.py
│   └── simulation.txt
├── api.py
│  
└── README.md
```

## Getting Started

This section will help you set up and run the project on your local machine.

### Prerequisites

- Node.js (for the front-end)
- Python (for the back-end)
- pip (Python package manager)

### Installing Dependencies

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/0-yibai/psa_hack_2024.git
   cd psa_hack_2024
   ```

2. Install dependencies for the front-end:

   ```bash
   cd prediction_app
   npm install
   ```

3. Install dependencies for the back-end:

   ```bash
   cd ..
   pip install -r requirements.txt
   ```

### Running the Application

#### Running the Front-End

1. Navigate to the `prediction_app` directory:

   ```bash
   cd prediction_app
   ```

2. Start the development server:

   ```bash
   npm start
   ```

   The app should now be running on `http://localhost:3000`.

#### Running the Back-End (API Server)

1. In another terminal window, navigate to the `psa_hack_2024` directory:

   ```bash
   cd psa_hack_2024
   ```

2. Run the model server:

   ```bash
   python api.py
   ```

   This will start the Flask API server on `http://127.0.0.1:5000`.

### Model Training

The predictive model was trained using a custom Graph Neural Network (GNN). Here's an overview of the training process:

1. **Dataset**:
   - The dataset used contains port waiting times for 50 ports over 1000 days.
   - The dataset is in CSV format with rows representing days and columns representing ports.
   - Another CSV file represents port relationships (e.g., travel likelihood between ports), which is used to define port-to-port connections for the GNN model.

2. **Model**:
   - The model used is a GNN with custom message-passing layers, which leverages the relationships between ports to improve the predictions.
   - Additionally, an LSTM layer is used to capture temporal dependencies in port waiting times (e.g., waiting times over the past 30 days).

3. **Training Process**:
   - The model was trained with a dataset of 1000 days. For each port, the model predicts the next day's waiting time based on the past 30 days.
   - During training, the input is a time-series of port waiting times, and the model uses the relationships between ports to capture interdependencies.
   - The output is the predicted waiting time for the next day.
   - The model was compiled using `Adam` optimizer and `mean squared error (MSE)` as the loss function.

4. **How to Train**:
   - You can re-train the model using the script `model_training.py` in the `scripts/` directory.
   - The training process looks like this:

     ```bash
     cd scripts
     python model_training.py
     ```

   - The trained model will be saved as `best_model_custom_gnn.keras`.

### Technologies Used

- **Front-End**:
  - React
  - CSS (for styling)
- **Back-End**:
  - Flask (for the API server)
  - TensorFlow (for the model)
  - Pandas, Numpy (for data handling)
  - Scikit-learn (for scaling the data)

### License

This project is licensed under the MIT License.
