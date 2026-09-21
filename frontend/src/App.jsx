import { useState } from "react";
import "./App.css";

function App() {
  const [backendStatus, setBackendStatus] = useState("Not checked");
  const [selectedFile, setSelectedFile] = useState(null);

  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadLoading, setUploadLoading] = useState(false);

  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);

  const [fileId, setFileId] = useState("");

  // --------------------------------------------------
  // CHECK BACKEND
  // --------------------------------------------------

  const checkBackend = async () => {
    setBackendStatus("Checking...");

    try {
      const response = await fetch("http://127.0.0.1:5000/health");

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const data = await response.json();

      if (data.status === "healthy") {
        setBackendStatus("🟢 Backend Connected");
      } else {
        setBackendStatus("🟡 Backend Responded");
      }
    } catch (error) {
      console.error(error);
      setBackendStatus("🔴 Backend Not Connected");
    }
  };

  // --------------------------------------------------
  // FILE SELECTION
  // --------------------------------------------------

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      setSelectedFile(null);
      setUploadStatus("");
      setFileId("");
      setPredictionResult(null);
      return;
    }

    setSelectedFile(file);
    setUploadStatus("");
    setFileId("");
    setPredictionResult(null);
  };

  // --------------------------------------------------
  // UPLOAD MRI
  // --------------------------------------------------

  const uploadMRI = async () => {
    if (!selectedFile) {
      setUploadStatus("❌ Please select an MRI file first.");
      return;
    }

    setUploadLoading(true);
    setUploadStatus("Uploading MRI...");
    setFileId("");
    setPredictionResult(null);

    try {
      const formData = new FormData();

      formData.append("file", selectedFile);

      const response = await fetch(
        "http://127.0.0.1:5000/upload",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          data.error || data.message || "MRI upload failed"
        );
      }

      setFileId(data.file_id);

      setUploadStatus(
        `✅ MRI uploaded successfully: ${data.filename}`
      );
    } catch (error) {
      console.error(error);

      setUploadStatus(
        `❌ Upload failed: ${error.message}`
      );
    } finally {
      setUploadLoading(false);
    }
  };

  // --------------------------------------------------
  // PREDICT MRI
  // --------------------------------------------------

  const predictMRI = async () => {
    if (!fileId) {
      setUploadStatus(
        "❌ Please upload the MRI before starting analysis."
      );
      return;
    }

    setPredictionLoading(true);
    setPredictionResult(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:5000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            file_id: fileId,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          data.error || "MRI prediction failed"
        );
      }

      setPredictionResult(data);
    } catch (error) {
      console.error(error);

      setPredictionResult({
        success: false,
        error: error.message,
      });
    } finally {
      setPredictionLoading(false);
    }
  };

  // --------------------------------------------------
  // FULL ANALYSIS
  // --------------------------------------------------

  const analyzeMRI = async () => {
    if (!selectedFile) {
      setUploadStatus(
        "❌ Please select an MRI file first."
      );
      return;
    }

    // First upload the MRI
    setUploadLoading(true);
    setUploadStatus("Uploading MRI...");
    setPredictionResult(null);
    setFileId("");

    try {
      const formData = new FormData();

      formData.append("file", selectedFile);

      const uploadResponse = await fetch(
        "http://127.0.0.1:5000/upload",
        {
          method: "POST",
          body: formData,
        }
      );

      const uploadData = await uploadResponse.json();

      if (!uploadResponse.ok || !uploadData.success) {
        throw new Error(
          uploadData.error ||
            uploadData.message ||
            "MRI upload failed"
        );
      }

      const newFileId = uploadData.file_id;

      setFileId(newFileId);

      setUploadStatus(
        `✅ MRI uploaded successfully: ${uploadData.filename}`
      );

      setUploadLoading(false);

      // ----------------------------------------------
      // Now call prediction
      // ----------------------------------------------

      setPredictionLoading(true);

      const predictionResponse = await fetch(
        "http://127.0.0.1:5000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            file_id: newFileId,
          }),
        }
      );

      const predictionData =
        await predictionResponse.json();

      if (
        !predictionResponse.ok ||
        !predictionData.success
      ) {
        throw new Error(
          predictionData.error ||
            "Prediction failed"
        );
      }

      setPredictionResult(predictionData);
    } catch (error) {
      console.error(error);

      setUploadStatus(
        `❌ Analysis failed: ${error.message}`
      );
    } finally {
      setUploadLoading(false);
      setPredictionLoading(false);
    }
  };

  // --------------------------------------------------
  // UI
  // --------------------------------------------------

  return (
    <div className="app">

      {/* HEADER */}

      <header className="header">
        <h1>Brain Tumor Detection System</h1>

        <p>
          Deep Learning-Based 3D Brain Tumor Detection,
          Segmentation and Quantitative Analysis
        </p>
      </header>

      {/* MAIN */}

      <main className="main">

        {/* BACKEND CARD */}

        <section className="card">

          <h2>Backend Connection</h2>

          <p className="description">
            Check whether the Flask backend is
            available.
          </p>

          <button
            onClick={checkBackend}
            className="button"
          >
            Check Backend
          </button>

          <div className="status">
            {backendStatus}
          </div>

        </section>

        {/* MRI UPLOAD CARD */}

        <section className="card">

          <h2>3D MRI Analysis</h2>

          <p className="description">
            Upload a 3D MRI scan in .nii or .nii.gz
            format.
          </p>

          <input
            type="file"
            accept=".nii,.nii.gz"
            onChange={handleFileChange}
          />

          {selectedFile && (
            <div className="selected-file">

              <strong>
                Selected MRI:
              </strong>

              <p>
                {selectedFile.name}
              </p>

              <p>
                Size:{" "}
                {(selectedFile.size / 1024).toFixed(2)}
                {" KB"}
              </p>

            </div>
          )}

          {/* UPLOAD BUTTON */}

          <button
            onClick={uploadMRI}
            disabled={
              !selectedFile ||
              uploadLoading
            }
            className="button"
          >
            {uploadLoading
              ? "Uploading..."
              : "Upload MRI"}
          </button>

          {/* ANALYZE BUTTON */}

          <button
            onClick={analyzeMRI}
            disabled={
              !selectedFile ||
              uploadLoading ||
              predictionLoading
            }
            className="button analyze-button"
          >
            {predictionLoading
              ? "Analyzing..."
              : "Analyze MRI"}
          </button>

          {/* UPLOAD STATUS */}

          {uploadStatus && (
            <div className="status">
              {uploadStatus}
            </div>
          )}

          {/* FILE ID */}

          {fileId && (
            <div className="file-id">

              <strong>
                File ID:
              </strong>

              <p>
                {fileId}
              </p>

            </div>
          )}

        </section>

        {/* PREDICTION RESULT */}

        {predictionResult && (
          <section className="card result-card">

            <h2>
              Analysis Result
            </h2>

            {predictionResult.success ? (
              <>
                <div className="result-row">

                  <span>
                    Tumor Present
                  </span>

                  <strong>
                    {predictionResult.prediction
                      .tumor_present
                      ? "Yes"
                      : "No"}
                  </strong>

                </div>

                <div className="result-row">

                  <span>
                    Tumor Type
                  </span>

                  <strong>
                    {
                      predictionResult
                        .prediction
                        .tumor_type
                    }
                  </strong>

                </div>

                <div className="result-row">

                  <span>
                    Confidence
                  </span>

                  <strong>
                    {(
                      predictionResult
                        .prediction
                        .confidence * 100
                    ).toFixed(2)}
                    %
                  </strong>

                </div>

                <div className="result-note">

                  ⚠️ Current result is from the
                  mock prediction endpoint. It is
                  not a real medical diagnosis.

                </div>
              </>
            ) : (
              <div className="error">
                ❌{" "}
                {predictionResult.error}
              </div>
            )}

          </section>
        )}

      </main>

    </div>
  );
}

export default App;