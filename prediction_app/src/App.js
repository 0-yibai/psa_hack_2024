// src/App.js

import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [newDayData, setNewDayData] = useState(Array(50).fill(''));
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState('');

  const handleInputChange = (index, value) => {
    const updatedData = [...newDayData];
    updatedData[index] = value;
    setNewDayData(updatedData);
  };

  const handleSubmit = async () => {
    try {
      setError('');
      setPredictions(null);

      // Convert input data to numbers
      const inputData = newDayData.map((value) => parseFloat(value));

      // Validate input data
      if (inputData.some(isNaN)) {
        setError('Please enter valid numerical values for all ports.');
        return;
      }

      // Send POST request to the API
      const response = await axios.post('http://127.0.0.1:5000/predict', {
        new_day_data: inputData,
      });

      setPredictions(response.data.predicted_waiting_times);
    } catch (err) {
      console.error(err);
      setError('An error occurred while making the prediction.');
    }
  };

  return (
    <div className="App">
      <h1>Port Waiting Time Prediction</h1>
      <p>Enter the waiting times for the new day for each port:</p>
      <div>
        {newDayData.map((value, index) => (
          <div key={index}>
            <label>
              Port {index + 1}:
              <input
                type="number"
                value={value}
                onChange={(e) => handleInputChange(index, e.target.value)}
              />
            </label>
          </div>
        ))}
      </div>
      <button onClick={handleSubmit}>Predict Next Day</button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {predictions && (
        <div>
          <h2>Predicted Waiting Times for the Next Day:</h2>
          <ul>
            {predictions.map((value, index) => (
              <li key={index}>
                Port {index + 1}: {value.toFixed(2)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
