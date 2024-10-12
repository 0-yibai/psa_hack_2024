import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [newDayData, setNewDayData] = useState(Array(50).fill('')); // No default value
  const [predictions, setPredictions] = useState(Array(50).fill(null)); // Empty prediction state for each port
  const [error, setError] = useState('');
  const [currentTime, setCurrentTime] = useState(new Date()); // Current date and time
  const [predictionDate, setPredictionDate] = useState(new Date()); // Date to predict (starting tomorrow)
  const [average, setAverage] = useState(null); // Average waiting time

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
      const predictedValues = response.data.predicted_waiting_times;
      setPredictions(predictedValues);

      // Use predicted values as the new inputs for the next round
      setNewDayData(predictedValues.map((pred) => pred.toFixed(2)));

      // Calculate the average of the predicted values
      const avg = predictedValues.reduce((sum, val) => sum + val, 0) / predictedValues.length;
      setAverage(avg);

      // Update prediction date to the next day (increment by 1 day)
      const nextDate = new Date(predictionDate);
      nextDate.setDate(predictionDate.getDate() + 1);
      setPredictionDate(nextDate);
    } catch (err) {
      console.error(err);
      setError('An error occurred while making the prediction.');
    }
  };

  const getBoxColor = (value) => {
    if (average === null) return '';

    if (value <= average * 0.5) {
      return 'deep-green'; // Way below average
    } else if (value < average) {
      return 'light-green'; // Below average
    } else if (value >= average * 1.5) {
      return 'deep-red'; // Way above average
    } else if (value > average) {
      return 'light-red'; // Above average
    } else {
      return ''; // Default color
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1 className="App-title">Port Waiting Time Prediction</h1>
        <p className="current-date">Current Date and Time: {currentTime.toLocaleDateString()} {currentTime.toLocaleTimeString()}</p>
        <p className="prediction-date">Predicting for: {predictionDate.toLocaleDateString()}</p>
        {average !== null && (
          <p className="average-waiting-time">
            Average Waiting Time: {average.toFixed(2)}
          </p>
        )}
      </header>

      <div className="form-container">
        {newDayData.map((value, index) => (
          <div key={index} className={`port-input ${getBoxColor(parseFloat(value))}`}>
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
