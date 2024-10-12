import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css'; // Import custom CSS for styling

function App() {
  const [newDayData, setNewDayData] = useState(Array(50).fill('')); // No default value
  const [predictions, setPredictions] = useState(Array(50).fill(null)); // Empty prediction state for each port
  const [error, setError] = useState('');
  const [currentTime, setCurrentTime] = useState(new Date()); // Current date and time
  const [predictionDate, setPredictionDate] = useState(new Date()); // Date to predict (starting tomorrow)

  // Update current time every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer); // Cleanup the timer on unmount
  }, []);

  // Set predictionDate to start as tomorrow
  useEffect(() => {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1); // Start from tomorrow
    setPredictionDate(tomorrow);
  }, []);

  const handleInputChange = (index, value) => {
    const updatedData = [...newDayData];
    updatedData[index] = value;
    setNewDayData(updatedData);
  };

  const handleSubmit = async () => {
    try {
      setError('');
      setPredictions(Array(50).fill(null)); // Clear previous predictions

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

      // Update the predictions state with the result
      setPredictions(response.data.predicted_waiting_times);

      // Use predicted values as the new inputs for next round
      setNewDayData(response.data.predicted_waiting_times.map((pred) => pred.toFixed(2)));

      // Update prediction date to the next day (increment by 1 day)
      const nextDate = new Date(predictionDate);
      nextDate.setDate(predictionDate.getDate() + 1);
      setPredictionDate(nextDate);
    } catch (err) {
      console.error(err);
      setError('An error occurred while making the prediction.');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1 className="App-title">Port Waiting Time Prediction</h1>
        <p className="current-date">Current Date and Time: {currentTime.toLocaleDateString()} {currentTime.toLocaleTimeString()}</p>
        <p className="prediction-date">Predicting for: {predictionDate.toLocaleDateString()}</p>
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
              placeholder="Enter waiting time"
            />
            {/* Display prediction beside the input field */}
            {predictions[index] !== null && (
              <span className="prediction-result">
                Prediction: {predictions[index].toFixed(2)}
              </span>
            )}
          </div>
        ))}
      </div>

      <button className="submit-button" onClick={handleSubmit}>
        Predict Next Day
      </button>

      {error && <p className="error-message">{error}</p>}
    </div>
  );
}

export default App;
