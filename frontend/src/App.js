import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setPrediction(data.prediction);
    } catch (error) {
      console.error('Error:', error);
      setPrediction('Error occurred during prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>CIFAR-10 Image Classifier</h1>
        <form onSubmit={handleSubmit}>
          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            accept="image/*"
            className="file-input"
          />
          <button type="submit" disabled={!file || loading}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>
        {prediction && (
          <div className="prediction">
            <h2>Prediction Result:</h2>
            <p>{prediction}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
