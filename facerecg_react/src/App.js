import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import Webcam from 'react-webcam';
import './App.css';

const API_URL = 'http://127.0.0.1:8000';

function App() {
  const [activeTab, setActiveTab] = useState('recognize');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [registerFile, setRegisterFile] = useState(null);
  const [registerPreviewUrl, setRegisterPreviewUrl] = useState('');
  const [personName, setPersonName] = useState('');
  const [databasePath, setDatabasePath] = useState('face_database.npz');
  const [datasetDir, setDatasetDir] = useState('./dataset/train');
  const [outputPath, setOutputPath] = useState('face_database.npz');
  const [threshold, setThreshold] = useState('0.75');
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [webcamEnabled, setWebcamEnabled] = useState(false);
  const webcamRef = useRef(null);

  useEffect(() => {
    checkHealth();
    fetchMetrics('face_database.npz');
  }, []);

  useEffect(() => {
    if (databasePath) {
      fetchMetrics(databasePath);
    }
  }, [databasePath]);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`);
      setHealth(response.data);
    } catch (error) {
      setHealth({
        status: 'error',
        message: 'Backend is not running. Start it with: python -m uvicorn face_api:app --host 0.0.0.0 --port 8000',
      });
    }
  };

  const fetchMetrics = async (database) => {
    try {
      const response = await axios.get(`${API_URL}/metrics`, {
        params: { database },
      });
      setMetrics(response.data);
    } catch (error) {
      setMetrics(null);
    }
  };

  const updatePreview = (file, currentUrl, setter) => {
    if (currentUrl) {
      URL.revokeObjectURL(currentUrl);
    }

    if (!file) {
      setter('');
      return;
    }

    setter(URL.createObjectURL(file));
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    setSelectedFile(file || null);
    setResult(null);
    updatePreview(file || null, previewUrl, setPreviewUrl);
  };

  const handleRegisterFileChange = (event) => {
    const file = event.target.files?.[0];
    setRegisterFile(file || null);
    updatePreview(file || null, registerPreviewUrl, setRegisterPreviewUrl);
  };

  const handleRecognize = async (fileOverride = null) => {
    const file = fileOverride || selectedFile;
    if (!file) {
      alert('Please select an image first');
      return;
    }

    setIsLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file, file.name || 'capture.jpg');
    formData.append('database', databasePath);
    formData.append('threshold', threshold);

    try {
      const response = await axios.post(`${API_URL}/recognize`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (error) {
      setResult({
        success: false,
        message: error.response?.data?.detail || 'Recognition request failed',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const captureFromWebcam = async () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (!imageSrc) {
      alert('Webcam is not ready');
      return;
    }

    const response = await fetch(imageSrc);
    const blob = await response.blob();
    const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });

    setSelectedFile(file);
    updatePreview(file, previewUrl, setPreviewUrl);
    await handleRecognize(file);
  };

  const handleRegisterFace = async () => {
    if (!registerFile) {
      alert('Please upload an image for registration');
      return;
    }

    if (!personName.trim()) {
      alert('Please enter a person name');
      return;
    }

    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', registerFile, registerFile.name);
    formData.append('person_name', personName);
    formData.append('dataset_dir', datasetDir);
    formData.append('output', outputPath);

    try {
      const response = await axios.post(`${API_URL}/register`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      alert(`${response.data.message}: ${response.data.person_name}`);
      setPersonName('');
      setRegisterFile(null);
      updatePreview(null, registerPreviewUrl, setRegisterPreviewUrl);
      setDatabasePath(outputPath);
      fetchMetrics(outputPath);
    } catch (error) {
      alert(error.response?.data?.detail || 'Face registration failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBuildDatabase = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post(`${API_URL}/build_database`, {
        dataset_dir: datasetDir,
        output: outputPath,
      });
      alert(`${response.data.message}. Generated ${response.data.num_identities} identity templates.`);
      setDatabasePath(outputPath);
    } catch (error) {
      alert(error.response?.data?.detail || 'Database build failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Face Recognition</h1>
      </header>

      <div className="status-banner">
        <strong>Backend status:</strong>
        <span className={health?.status === 'ok' ? 'ok-text' : 'error-text'}>
          {health?.message || 'Checking...'}
        </span>
        <button type="button" onClick={checkHealth}>Check again</button>
      </div>

      <div className="tabs">
        <button className={activeTab === 'recognize' ? 'active' : ''} onClick={() => setActiveTab('recognize')}>
          Recognize
        </button>
        <button className={activeTab === 'register' ? 'active' : ''} onClick={() => setActiveTab('register')}>
          Register Face
        </button>
        <button className={activeTab === 'build' ? 'active' : ''} onClick={() => setActiveTab('build')}>
          Build Database
        </button>
      </div>

      <div className="content">
        {activeTab === 'recognize' && (
          <div className="recognize-tab">
            <label className="field-label" htmlFor="databasePath">Database path</label>
            <input
              id="databasePath"
              type="text"
              className="text-input"
              value={databasePath}
              onChange={(e) => setDatabasePath(e.target.value)}
            />

            <label className="field-label" htmlFor="threshold">Threshold</label>
            <input
              id="threshold"
              type="number"
              step="0.01"
              min="0"
              max="1"
              className="text-input"
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
            />

            <label className="field-label" htmlFor="imageInput">Upload image</label>
            <input id="imageInput" type="file" accept="image/*" onChange={handleFileChange} />

            <div className="action-row">
              <button type="button" onClick={() => handleRecognize()} disabled={isLoading}>
                {isLoading ? 'Recognizing...' : 'Recognize'}
              </button>
              <button type="button" onClick={() => setWebcamEnabled((prev) => !prev)}>
                {webcamEnabled ? 'Disable Webcam' : 'Enable Webcam'}
              </button>
            </div>

            {webcamEnabled && (
              <div className="webcam-section">
                <Webcam
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={{ width: 640, height: 480, facingMode: 'user' }}
                  className="webcam-view"
                />
                <button type="button" onClick={captureFromWebcam} disabled={isLoading}>
                  Capture and Recognize
                </button>
              </div>
            )}

            {previewUrl && (
              <div className="image-preview">
                <h3>Selected Image</h3>
                <img src={previewUrl} alt="preview" style={{ maxWidth: '420px', width: '100%' }} />
              </div>
            )}

            {result && (
              <div className="results">
                <h3>Recognition Result</h3>
                {result.success ? (
                  <>
                    <div className={`result-chip ${result.is_known ? 'known' : 'unknown'}`}>
                      {result.is_known ? 'Known' : 'Stranger'}
                    </div>
                    <p><strong>Prediction:</strong> {result.predicted_identity}</p>
                    <p><strong>Best match:</strong> {result.best_match}</p>
                    <p><strong>Similarity:</strong> {result.similarity}</p>
                    <p><strong>Threshold:</strong> {result.threshold}</p>
                    {result.face_box && (
                      <p><strong>Face box:</strong> [{result.face_box.join(', ')}]</p>
                    )}
                  </>
                ) : (
                  <p className="error-text">{result.message || 'No valid face detected'}</p>
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'register' && (
          <div className="register-tab">
            <h3>Register a new face</h3>

            <label className="field-label" htmlFor="personName">Person name</label>
            <input
              id="personName"
              type="text"
              className="text-input"
              value={personName}
              onChange={(e) => setPersonName(e.target.value)}
            />

            <label className="field-label" htmlFor="registerDatasetDir">Dataset directory</label>
            <input
              id="registerDatasetDir"
              type="text"
              className="text-input"
              value={datasetDir}
              onChange={(e) => setDatasetDir(e.target.value)}
            />

            <label className="field-label" htmlFor="registerOutputPath">Output database file</label>
            <input
              id="registerOutputPath"
              type="text"
              className="text-input"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
            />

            <label className="field-label" htmlFor="registerImageInput">Upload image</label>
            <input id="registerImageInput" type="file" accept="image/*" onChange={handleRegisterFileChange} />

            {registerPreviewUrl && (
              <div className="image-preview">
                <h3>Registration Preview</h3>
                <img src={registerPreviewUrl} alt="registration preview" style={{ maxWidth: '420px', width: '100%' }} />
              </div>
            )}

            <button type="button" onClick={handleRegisterFace} disabled={isLoading}>
              {isLoading ? 'Registering...' : 'Register Face'}
            </button>
          </div>
        )}

        {activeTab === 'build' && (
          <div className="register-tab">
            <h3>Build a database from the server dataset</h3>

            <label className="field-label" htmlFor="datasetDir">Dataset directory</label>
            <input
              id="datasetDir"
              type="text"
              className="text-input"
              value={datasetDir}
              onChange={(e) => setDatasetDir(e.target.value)}
            />

            <label className="field-label" htmlFor="outputPath">Output database file</label>
            <input
              id="outputPath"
              type="text"
              className="text-input"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
            />

            <button type="button" onClick={handleBuildDatabase} disabled={isLoading}>
              {isLoading ? 'Building...' : 'Build Database'}
            </button>
          </div>
        )}
      </div>

      <div className="metrics-info">
        <h3>System Metrics</h3>
        <div className="metrics-grid">
          <div className="metric">
            <span className="metric-value">{metrics?.num_identities ?? '--'}</span>
            <span className="metric-label">Registered Identities</span>
          </div>
          <div className="metric">
            <span className="metric-value">{metrics?.detector ?? '--'}</span>
            <span className="metric-label">Face Detector</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
