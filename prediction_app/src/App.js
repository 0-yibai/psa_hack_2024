import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // Import custom CSS for styling

function App() {
  const [newDayData, setNewDayData] = useState(Array(50).fill(1000)); // Default value of 1000 for each port
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
      <header className="App-header">
        <h1 className="App-title">Port Waiting Time Prediction</h1>
        <p className="App-description">
          Enter the waiting times for the new day for each port (default is 1000).
        </p>
      </header>
      
      <div className="form-container">
        {newDayData.map((value, index) => (
          <div key={index} className="port-input">
            <label className="port-label">Port {index + 1}:</label>
            <input
              type="number"
              value={value}
              onChange={(e) => handleInputChange(index, e.target.value)}
              className="input-field"
            />
          </div>
        ))}
      </div>

      <button className="submit-button" onClick={handleSubmit}>
        Predict Next Day
      </button>
      
      {error && <p className="error-message">{error}</p>}
      
      {predictions && (
        <div className="predictions-container">
          <h2 className="predictions-title">Predicted Waiting Times for the Next Day:</h2>
          <ul className="predictions-list">
            {predictions.map((value, index) => (
              <li key={index} className="prediction-item">
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
