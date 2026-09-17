import { useState } from "react";
import "./App.css";

function App() {
  // Backend status
  const [backendStatus, setBackendStatus] = useState("Not checked");

  // Backend health-check loading
  const [backendLoading, setBackendLoading] = useState(false);

  // Selected MRI file
  const [selectedFile, setSelectedFile] = useState(null);

  // MRI upload status
  const [uploadStatus, setUploadStatus] = useState("No MRI uploaded");

  // MRI upload loading
  const [uploadLoading, setUploadLoading] = useState(false);

  // File ID returned by Flask
  const [fileId, setFileId] = useState("");

  // -----------------------------------------
  // CHECK FLASK BACKEND
  // -----------------------------------------

  const checkBackend = async () => {
    setBackendLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:5000/health"
      );

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const data = await response.json();

      if (data.status === "healthy") {
        setBackendStatus("🟢 Backend Connected");
      } else {
        setBackendStatus("🟡 Backend responded");
      }
    } catch (error) {
      console.error(error);
      setBackendStatus("🔴 Backend Not Connected");
    } finally {
      setBackendLoading(false);
    }
  };

  // -----------------------------------------
  // HANDLE FILE SELECTION
  // -----------------------------------------

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      setSelectedFile(null);
      setUploadStatus("No MRI selected");
      setFileId("");
      return;
    }

    setSelectedFile(file);
    setUploadStatus(`Selected: ${file.name}`);
    setFileId("");
  };

  // -----------------------------------------
  // UPLOAD MRI TO FLASK
  // -----------------------------------------

  const analyzeMRI = async () => {
    // Make sure a file has been selected
    if (!selectedFile) {
      setUploadStatus("🔴 Please select an MRI file first.");
      return;
    }

    // Make sure the file has a valid extension
    const fileName = selectedFile.name.toLowerCase();

    if (
      !fileName.endsWith(".nii") &&
      !fileName.endsWith(".nii.gz")
    ) {
      setUploadStatus(
        "🔴 Invalid file. Please select a .nii or .nii.gz file."
      );
      return;
    }

    setUploadLoading(true);
    setUploadStatus("Uploading MRI...");
    setFileId("");

    try {
      // Create multipart form data
      const formData = new FormData();

      // IMPORTANT:
      // "file" must match Flask request.files["file"]
      formData.append("file", selectedFile);

      // Send MRI to Flask
      const response = await fetch(
        "http://127.0.0.1:5000/upload",
        {
          method: "POST",
          body: formData,
        }
      );

      // Read Flask response
      const data = await response.json();

      // Handle Flask error
      if (!response.ok) {
        throw new Error(
          data.error || "MRI upload failed."
        );
      }

      // Successful upload
      setUploadStatus(
        `🟢 ${data.message}`
      );

      setFileId(data.file_id);

    } catch (error) {
      console.error("MRI upload error:", error);

      setUploadStatus(
        `🔴 ${error.message}`
      );

    } finally {
      setUploadLoading(false);
    }
  };

  return (
    <div className="app">

      {/* -------------------------------- */}
      {/* HEADER                           */}
      {/* -------------------------------- */}

      <header className="header">

        <h1>
          Brain Tumor Detection System
        </h1>

        <p>
          Deep Learning-Based 3D Brain Tumor Detection,
          Segmentation and Quantitative Analysis
        </p>

      </header>


      {/* -------------------------------- */}
      {/* MAIN                             */}
      {/* -------------------------------- */}

      <main className="main">

        {/* ============================== */}
        {/* BACKEND CONNECTION              */}
        {/* ============================== */}

        <section className="card">

          <h2>
            Backend Connection
          </h2>

          <p className="description">
            Check whether the Flask backend is available.
          </p>

          <button
            onClick={checkBackend}
            disabled={backendLoading}
          >
            {backendLoading
              ? "Checking..."
              : "Check Backend"}
          </button>

          <div className="status">
            {backendStatus}
          </div>

        </section>


        {/* ============================== */}
        {/* MRI UPLOAD                      */}
        {/* ============================== */}

        <section className="card">

          <h2>
            3D MRI Analysis
          </h2>

          <p className="description">
            Upload a 3D MRI scan in .nii or .nii.gz format.
          </p>


          {/* File input */}

          <input
            type="file"
            accept=".nii,.nii.gz"
            onChange={handleFileChange}
          />


          {/* Analyze / Upload button */}

          <button
            className="analyze-button"
            onClick={analyzeMRI}
            disabled={uploadLoading}
          >
            {uploadLoading
              ? "Uploading..."
              : "Analyze MRI"}
          </button>


          {/* Upload status */}

          <div className="status">
            {uploadStatus}
          </div>


          {/* File ID */}

          {fileId && (
            <div className="file-id">
              <strong>File ID:</strong>

              <br />

              {fileId}
            </div>
          )}

        </section>

      </main>

    </div>
  );
}

export default App;