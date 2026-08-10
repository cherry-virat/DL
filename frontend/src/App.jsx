import { useState } from "react";
import "./App.css";

function App() {
  const [backendStatus, setBackendStatus] = useState("Not checked");
  const [loading, setLoading] = useState(false);

  const checkBackend = async () => {
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/health");

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const data = await response.json();

      if (data.status === "healthy") {
        setBackendStatus("🟢 Backend Connected");
      } else {
        setBackendStatus("🟠 Backend responded");
      }
    } catch (error) {
      console.error(error);
      setBackendStatus("🔴 Backend Not Connected");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Brain Tumor Detection System</h1>

        <p>
          Deep Learning-Based 3D Brain Tumor Detection,
          Segmentation and Quantitative Analysis
        </p>
      </header>

      <main className="main">
        <section className="card">
          <h2>Backend Connection</h2>

          <p className="description">
            Check whether the Flask backend is available.
          </p>

          <button onClick={checkBackend} disabled={loading}>
            {loading ? "Checking..." : "Check Backend"}
          </button>

          <div className="status">
            {backendStatus}
          </div>
        </section>

        <section className="card">
          <h2>3D MRI Analysis</h2>

          <p className="description">
            Upload a 3D MRI scan in .nii or .nii.gz format.
          </p>

          <input
            type="file"
            accept=".nii,.nii.gz"
          />

          <button className="analyze-button">
            Analyze MRI
          </button>
        </section>
      </main>
    </div>
  );
}

export default App;