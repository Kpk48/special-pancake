"use client";

import {
  BrainCircuit,
  FileImage,
  Loader2,
  UploadCloud,
  Camera
} from "lucide-react";
import { FormEvent, useMemo, useRef, useState } from "react";

type Prediction = {
  label: string;
  probabilities: Record<string, number>;
};

const classes = [
  "cardboard",
  "glass",
  "metal",
  "organic",
  "paper",
  "plastic",
  "textile",
  "battery",
  "wood",
  "ceramic",
  "nylon"
];
export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [livePrediction, setLivePrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isLiveAnalyzing, setIsLiveAnalyzing] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const liveCanvasRef = useRef<HTMLCanvasElement>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const liveAnalysisRef = useRef<boolean>(false);

  const sortedProbabilities = useMemo(() => {
    if (!prediction) {
      return [];
    }
    return Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1]);
  }, [prediction]);

  async function startCamera() {
    try {
      setError("");
      setPrediction(null);
      setLivePrediction(null);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
        setIsLiveAnalyzing(true);
        liveAnalysisRef.current = true;
        startLiveClassification();
      }
    } catch (err) {
      setError("Unable to access camera. Please check permissions.");
    }
  }

  async function startLiveClassification() {
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    
    frameIntervalRef.current = setInterval(async () => {
      if (!liveAnalysisRef.current || !videoRef.current || !liveCanvasRef.current) return;
      
      try {
        const context = liveCanvasRef.current.getContext("2d");
        if (!context) return;

        liveCanvasRef.current.width = videoRef.current.videoWidth;
        liveCanvasRef.current.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0);

        liveCanvasRef.current.toBlob(async (blob) => {
          if (!blob) return;
          
          const formData = new FormData();
          formData.append("image", blob, "frame.jpg");

          try {
            const response = await fetch("/api/predict", {
              method: "POST",
              body: formData
            });
            const data = await response.json();
            if (response.ok) {
              setLivePrediction(data as Prediction);
            }
          } catch (error) {
            // Silently fail for live prediction
          }
        }, "image/jpeg", 0.7);
      } catch (error) {
        // Silently fail for live prediction
      }
    }, 500); // Analyze every 500ms
  }

  async function capturePhoto() {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext("2d");
      if (context) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0);
        
        canvasRef.current.toBlob((blob) => {
          if (blob) {
            const capturedFile = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
            setFile(capturedFile);
            stopCameraAndClassify();
          }
        }, "image/jpeg", 0.9);
      }
    }
  }

  function stopCameraAndClassify() {
    stopCamera();
    // Automatically submit for classification after a short delay
    setTimeout(() => {
      const form = document.getElementById("classifyForm") as HTMLFormElement;
      if (form) {
        form.dispatchEvent(new Event("submit", { bubbles: true }));
      }
    }, 100);
  }

  function stopCamera() {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    liveAnalysisRef.current = false;
    setIsLiveAnalyzing(false);
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      setIsCameraActive(false);
    }
    setLivePrediction(null);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setPrediction(null);

    if (!file) {
      setError("Please select an image file to classify.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);
    setIsLoading(true);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error || "Classification failed. Please try again.");
      }

      setPrediction(payload as Prediction);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Classification failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <>
      <header className="header">
        <div className="shell headerContent">
          <h1 className="headerTitle">Waste Classification System</h1>
          <p className="headerSubtitle">AI-powered waste material recognition</p>
        </div>
      </header>

      <main className="shell">
        <section className="hero">
          <div className="heroCopy">
            <h2>Intelligent Waste Sorting</h2>
            <p>
              Upload an image to instantly identify waste materials. Our machine learning model
              recognizes 11 different waste categories including cardboard, glass, metal, plastic,
              paper, organic waste, textiles, ceramic, battery, wood, and nylon.
            </p>
            <div className="featureList">
              <div className="feature">
                <FileImage size={20} />
                <span>Upload or Capture Images</span>
              </div>
              <div className="feature">
                <BrainCircuit size={20} />
                <span>Machine Learning Based</span>
              </div>
            </div>
          </div>
        </section>

        <section className="workspace" id="classifier">
          <div className="glassCard classifierCard">
            <div className="cardHeader">
              <h3>Upload Image</h3>
              <span className="formatHint">JPG, PNG, PPM, RAW</span>
            </div>

            <form onSubmit={handleSubmit} className="uploadForm" id="classifyForm">
              {!isCameraActive ? (
                <>
                  <label className="dropZone">
                    <UploadCloud size={40} />
                    <span>{file ? file.name : "Select or drag image"}</span>
                    <small>Supports JPG, JPEG, PNG, PPM, and RAW formats</small>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".ppm,.jpg,.jpeg,.png,.raw,image/*"
                      onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                    />
                  </label>
                  <div className="buttonGroup">
                    <button className="predictButton" type="submit" disabled={isLoading || !file}>
                      {isLoading ? <Loader2 className="spin" size={18} /> : <BrainCircuit size={18} />}
                      {isLoading ? "Analyzing..." : "Classify"}
                    </button>
                    <button 
                      type="button" 
                      className="cameraButton" 
                      onClick={startCamera}
                      disabled={isLoading}
                    >
                      <Camera size={18} />
                      Open Camera
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <div className="cameraContainer" onClick={capturePhoto}>
                    <video 
                      ref={videoRef} 
                      autoPlay 
                      playsInline 
                      className="videoStream"
                    />
                    <canvas ref={liveCanvasRef} className="hiddenCanvas" />
                    
                    {/* Live Prediction Overlay */}
                    {livePrediction && (
                      <div className="liveOverlay">
                        <div className="livePredictionBox">
                          <div className="livePredictionLabel">Live Detection</div>
                          <div className="livePredictionResult">{livePrediction.label}</div>
                          <div className="liveConfidence">
                            {Math.round((Math.max(...Object.values(livePrediction.probabilities)) || 0) * 100)}% Confidence
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* Click to Capture Instruction */}
                    <div className="captureInstruction">
                      <div className="instructionBox">
                        <Camera size={24} />
                        <span>Click to Capture & Classify</span>
                      </div>
                    </div>
                  </div>
                  <div className="buttonGroup">
                    <button 
                      type="button" 
                      className="captureButton"
                      onClick={capturePhoto}
                    >
                      <Camera size={18} />
                      Capture Photo
                    </button>
                    <button 
                      type="button" 
                      className="secondaryButton"
                      onClick={stopCamera}
                    >
                      Cancel
                    </button>
                  </div>
                </>
              )}
            </form>

            {error && <div className="alert alertError">{error}</div>}

            {prediction && (
              <div className="resultPanel">
                <div className="resultHeader">
                  <div>
                    <p className="resultLabel">Predicted Category</p>
                    <h3>{prediction.label}</h3>
                  </div>
                </div>
                <div className="confidenceSection">
                  <p className="confidenceLabel">Classification Confidence</p>
                  <div className="confidenceList">
                    {sortedProbabilities.map(([name, score]) => (
                      <div className="confidenceRow" key={name}>
                        <span className="categoryName">{name}</span>
                        <div className="track">
                          <span className="confidence" style={{ width: `${Math.round(score * 100)}%` }} />
                        </div>
                        <strong className="percentage">{Math.round(score * 100)}%</strong>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <aside className="sidePanel">
            <div className="glassCard infoCard">
              <h3>Supported Categories</h3>
              <ul className="categoryList">
                {classes.map((cls) => (
                  <li key={cls}>{cls}</li>
                ))}
              </ul>
            </div>

            <div className="glassCard instructionsCard">
              <h3>How to Use</h3>
              <ol className="instructionsList">
                <li>Upload an image (JPG, JPEG, PNG, PPM, RAW) or capture with camera</li>
                <li>Click the upload area or use the camera button</li>
                <li>Click "Classify" to analyze the image</li>
                <li>View the predicted category and confidence scores</li>
              </ol>
            </div>
          </aside>
        </section>
      </main>

      <footer className="footer">
        <div className="shell">
          <p>&copy; 2024 Waste Classification System. AI-powered sustainability initiative.</p>
        </div>
      </footer>
    </>
  );
}
