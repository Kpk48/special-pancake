"use client";

import {
  BrainCircuit,
  FileImage,
  Loader2,
  UploadCloud
} from "lucide-react";
import { FormEvent, useMemo, useState } from "react";

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
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const sortedProbabilities = useMemo(() => {
    if (!prediction) {
      return [];
    }
    return Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1]);
  }, [prediction]);

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
                <span>Fast & Accurate Classification</span>
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
              <span className="formatHint">PPM format</span>
            </div>

            <form onSubmit={handleSubmit} className="uploadForm">
              <label className="dropZone">
                <UploadCloud size={40} />
                <span>{file ? file.name : "Select or drag image"}</span>
                <small>PPM format images are supported</small>
                <input
                  type="file"
                  accept=".ppm"
                  onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                />
              </label>
              <button className="predictButton" type="submit" disabled={isLoading || !file}>
                {isLoading ? <Loader2 className="spin" size={18} /> : <BrainCircuit size={18} />}
                {isLoading ? "Analyzing..." : "Classify"}
              </button>
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
                <li>Prepare a waste image in PPM format</li>
                <li>Click the upload area to select your file</li>
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
