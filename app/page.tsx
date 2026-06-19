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
    if (!prediction) return [];
    return Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1]);
  }, [prediction]);

  async function startCamera() {
    try {
      setError("");
      setPrediction(null);
      setLivePrediction(null);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 960 },
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);

        await new Promise<void>((resolve) => {
          const video = videoRef.current!;
          if (video.readyState >= 2) {
            resolve();
          } else {
            video.addEventListener("canplay", () => resolve(), { once: true });
          }
        });

        await videoRef.current.play();
        setIsLiveAnalyzing(true);
        liveAnalysisRef.current = true;
        startLiveClassification();
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      setError(`Camera access failed: ${errorMsg}`);
      setIsCameraActive(false);
    }
  }

  function startLiveClassification() {
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);

    frameIntervalRef.current = setInterval(async () => {
      if (!liveAnalysisRef.current || !videoRef.current || !liveCanvasRef.current) return;

      try {
        if (videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) return;

        const context = liveCanvasRef.current.getContext("2d");
        if (!context) return;

        liveCanvasRef.current.width = videoRef.current.videoWidth;
        liveCanvasRef.current.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0);

        liveCanvasRef.current.toBlob(
          async (blob) => {
            if (!blob) return;
            const formData = new FormData();
            formData.append("image", blob, "frame.jpg");
            try {
              const response = await fetch("/api/predict", {
                method: "POST",
                body: formData,
              });
              if (response.ok) {
                const data = await response.json();
                setLivePrediction(data as Prediction);
              }
            } catch {
              // silently continue on network errors
            }
          },
          "image/jpeg",
          0.8
        );
      } catch {
        // silently continue
      }
    }, 500);
  }

  async function capturePhoto() {
    if (!videoRef.current || !canvasRef.current) {
      setError("Failed to capture photo from camera.");
      return;
    }

    try {
      setIsLoading(true);
      setError("");
      setPrediction(null);

      const context = canvasRef.current.getContext("2d");
      if (!context) {
        setError("Failed to process camera capture.");
        setIsLoading(false);
        return;
      }

      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
      context.drawImage(videoRef.current, 0, 0);

      canvasRef.current.toBlob(
        async (blob) => {
          if (!blob) {
            setError("Failed to create image from camera capture.");
            setIsLoading(false);
            return;
          }

          try {
            const formData = new FormData();
            formData.append("image", blob, "camera-capture.jpg");

            const response = await fetch("/api/predict", {
              method: "POST",
              body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
              throw new Error(data.error || "Classification failed.");
            }

            setPrediction(data as Prediction);
            stopCamera();
          } catch (err) {
            setError(err instanceof Error ? err.message : "Classification failed. Please try again.");
          } finally {
            setIsLoading(false);
          }
        },
        "image/jpeg",
        0.95
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to capture photo.");
      setIsLoading(false);
    }
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
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
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
        body: formData,
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
                  {/* Camera viewport — no onClick here, only on the button below */}
                  <div className="cameraContainer">
                    <video
                      ref={videoRef}
                      autoPlay
                      muted
                      playsInline
                      className="videoStream"
                    />

                    {/* Hidden canvases */}
                    <canvas ref={canvasRef} className="hiddenCanvas" />
                    <canvas ref={liveCanvasRef} className="hiddenCanvas" />

                    {/* Live detection label overlay (top-left) */}
                    {isLiveAnalyzing && livePrediction && (
                      <div className="liveOverlay">
                        <div className="livePredictionBox">
                          <div className="livePredictionLabel">🎯 Live Detection</div>
                          <div className="livePredictionResult">
                            {livePrediction.label.toUpperCase()}
                          </div>
                          <div className="liveConfidence">
                            {Math.round(
                              (Math.max(...Object.values(livePrediction.probabilities)) || 0) * 100
                            )}% confidence
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Top-3 probability bars (bottom of video) */}
                    {isLiveAnalyzing && livePrediction && (
                      <div className="liveBottomBar">
                        {Object.entries(livePrediction.probabilities)
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 3)
                          .map(([name, score]) => (
                            <div key={name} className="liveBarRow">
                              <span className="liveBarLabel">{name}</span>
                              <div className="liveBarTrack">
                                <div
                                  className="liveBarFill"
                                  style={{ width: `${Math.round(score * 100)}%` }}
                                />
                              </div>
                              <span className="liveBarPct">{Math.round(score * 100)}%</span>
                            </div>
                          ))}
                      </div>
                    )}

                    {/* "Point at waste" hint — shown before first detection */}
                    {isLiveAnalyzing && !livePrediction && (
                      <div className="captureInstruction">
                        <div className="instructionBox">
                          <Camera size={24} />
                          <span>Point at waste to classify</span>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="buttonGroup">
                    <button
                      type="button"
                      className="captureButton"
                      onClick={capturePhoto}
                      disabled={isLoading}
                    >
                      {isLoading ? <Loader2 className="spin" size={18} /> : <Camera size={18} />}
                      {isLoading ? "Classifying..." : "Capture & Classify"}
                    </button>
                    <button
                      type="button"
                      className="secondaryButton"
                      onClick={stopCamera}
                      disabled={isLoading}
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
                          <span
                            className="confidence"
                            style={{ width: `${Math.round(score * 100)}%` }}
                          />
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