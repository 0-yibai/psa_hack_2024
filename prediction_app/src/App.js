import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css'; // Import custom CSS for styling

function App() {
  const generateRandomData = () => Array.from({ length: 50 }, () => Math.floor(Math.random() * 1001));
  const [newDayData, setNewDayData] = useState(generateRandomData());
  const [predictions, setPredictions] = useState(Array(50).fill(null));
  const [error, setError] = useState('');
  const [currentTime, setCurrentTime] = useState(new Date());
  const [predictionDate, setPredictionDate] = useState(new Date());
  const [average, setAverage] = useState(null);
  const [showWelcomePopup, setShowWelcomePopup] = useState(true); // Control the welcome pop-up visibility

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

      const inputData = newDayData.map((value) => parseFloat(value));
      if (inputData.some(isNaN)) {
        setError('Please enter valid numerical values for all ports.');
        return;
      }

      const response = await axios.post('http://127.0.0.1:5000/predict', {
        new_day_data: inputData,
      });

      const predictedValues = response.data.predicted_waiting_times;
      setPredictions(predictedValues);
      setNewDayData(predictedValues.map((pred) => pred.toFixed(2)));

      const avg = predictedValues.reduce((sum, val) => sum + val, 0) / predictedValues.length;
      setAverage(avg);

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
      return 'deep-green';
    } else if (value < average) {
      return 'light-green';
    } else if (value >= average * 1.5) {
      return 'deep-red';
    } else if (value > average) {
      return 'light-red';
    } else {
      return '';
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

      {showWelcomePopup && (
        <div className="welcome-popup">
          <div className="welcome-content">
            <h2>Welcome to the PSA ports waiting time prediction website!</h2>
            <h3>Please input waiting time of each port today and click "Predict Next Day" to view the out come!</h3>
            <button className="close-button" onClick={() => setShowWelcomePopup(false)}>Close</button>
          </div>
        </div>
      )}

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
