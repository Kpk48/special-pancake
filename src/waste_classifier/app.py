"""Local browser app for waste image prediction."""

from __future__ import annotations

import argparse
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .features import extract_features
from .model import load_model


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI-Based Waste Classification System</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f7f4; color: #182018; }}
    main {{ max-width: 860px; margin: 0 auto; padding: 40px 20px; }}
    h1 {{ font-size: 34px; margin: 0 0 10px; }}
    p {{ line-height: 1.55; }}
    form {{ margin-top: 28px; padding: 22px; border: 1px solid #d7dfd1; background: white; border-radius: 8px; }}
    input, button {{ font-size: 16px; }}
    button {{ margin-top: 16px; padding: 10px 16px; background: #206a3b; color: white; border: 0; border-radius: 6px; cursor: pointer; }}
    .result {{ margin-top: 22px; padding: 18px; background: #e9f4ea; border-left: 5px solid #206a3b; }}
    .error {{ margin-top: 22px; padding: 18px; background: #fff1f0; border-left: 5px solid #b42318; }}
    code {{ background: #eef1ec; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <main>
    <h1>AI-Based Waste Classification System</h1>
    <p>Upload a <code>.ppm</code> waste image and the trained model will classify it as cardboard, glass, metal, organic, paper, or plastic.</p>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept=".ppm" required>
      <br>
      <button type="submit">Classify Waste</button>
    </form>
    {message}
  </main>
</body>
</html>
"""


class WasteRequestHandler(BaseHTTPRequestHandler):
    model_path = Path("artifacts/waste_model.json")

    def do_GET(self) -> None:
        self._send_page("")

    def do_POST(self) -> None:
        try:
            data = self._read_uploaded_file()
            with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as handle:
                handle.write(data)
                temp_path = Path(handle.name)

            model = load_model(self.model_path)
            features = extract_features(temp_path)
            label = model.predict(features)
            probabilities = model.predict_proba(features)
            temp_path.unlink(missing_ok=True)
            confidence = ", ".join(f"{name}: {score:.2f}" for name, score in probabilities.items())
            self._send_page(f'<div class="result"><strong>Prediction:</strong> {label}<br><strong>Confidence:</strong> {confidence}</div>')
        except Exception as exc:
            self._send_page(f'<div class="error"><strong>Error:</strong> {exc}</div>')

    def _read_uploaded_file(self) -> bytes:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type or "boundary=" not in content_type:
            raise ValueError("Expected a multipart file upload")
        boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode("utf-8")
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        delimiter = b"--" + boundary
        for part in body.split(delimiter):
            if b'name="image"' not in part:
                continue
            header_end = part.find(b"\r\n\r\n")
            if header_end == -1:
                header_end = part.find(b"\n\n")
                separator_length = 2
            else:
                separator_length = 4
            if header_end == -1:
                raise ValueError("Could not parse uploaded file")
            file_bytes = part[header_end + separator_length:]
            return file_bytes.rstrip(b"\r\n-")
        raise ValueError("No image file field named 'image' was uploaded")

    def _send_page(self, message: str) -> None:
        body = HTML.format(message=message).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the waste classification web app.")
    parser.add_argument("--model", default="artifacts/waste_model.json", help="Model JSON path.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on.")
    args = parser.parse_args()

    WasteRequestHandler.model_path = Path(args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), WasteRequestHandler)
    print(f"Serving waste classifier at http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
