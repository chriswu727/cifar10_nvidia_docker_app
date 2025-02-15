import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [predictions, setPredictions] = useState([]);

  // Create preview when file is selected
  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);

    // Free memory when component unmounts
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const fetchPredictions = async () => {
    try {
      const response = await fetch('/api/predictions');
      const data = await response.json();
      setPredictions(data);
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
  };

  const handleShowHistory = () => {
    setShowHistory(!showHistory);
    if (!showHistory) {
      fetchPredictions();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>CIFAR-10 Image Classifier</h1>
        
        {/* File Upload */}
        <div className="upload-section">
          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            accept="image/*"
            className="file-input"
          />
        </div>

        {/* Buttons */}
        <div className="button-group">
          <button type="button" disabled={!file || loading} onClick={handleSubmit}>
            {loading ? 'Processing...' : 'Predict'}
          </button>
          <button 
            onClick={handleShowHistory} 
            className="history-button"
          >
            {showHistory ? 'Hide History' : 'Show History'}
          </button>
        </div>

        {/* Main Content Area */}
        <div className="main-content">
          {/* Image Preview */}
          {preview && (
            <div className="image-preview">
              <img src={preview} alt="Preview" />
            </div>
          )}

          {/* Prediction Result */}
          {prediction && (
            <div className="prediction">
              <h2>Prediction Result:</h2>
              <p>Class: {prediction.prediction}</p>
              <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
            </div>
          )}
        </div>

        {/* Prediction History Sidebar */}
        {showHistory && (
          <div className="prediction-history">
            <div className="history-header">
              <h2>Prediction History</h2>
              <button 
                onClick={fetchPredictions}
                className="refresh-button"
              >
                Refresh
              </button>
            </div>
            <div className="history-list">
              {predictions.map((pred, index) => (
                <div key={index} className="history-item">
                  <p>File: {pred.filename}</p>
                  <p>Prediction: {pred.prediction}</p>
                  <p>Confidence: {(pred.confidence * 100).toFixed(2)}%</p>
                  <p>Time: {new Date(pred.timestamp).toLocaleString()}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
