import sys
import os
import json
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.hierarchy import STAGE3_CLASSES

DEVICE = torch.device("cpu")
STAGE1 = None
STAGE2 = None
STAGE3 = None
STAGE1_THRESHOLD = 0.55
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def load_models():
    global STAGE1, STAGE2, STAGE3, STAGE1_THRESHOLD
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s1_path = os.path.join(root_dir, "artifacts", "hierarchical", "stage1_v2_relabeled.pt")
    s2_path = os.path.join(root_dir, "artifacts", "hierarchical", "stage2.pt")
    s3_path = os.path.join(root_dir, "artifacts", "hierarchical", "stage3.pt")
    calib_path = os.path.join(root_dir, "artifacts", "hierarchical", "stage1_calibration.json")
    
    if os.path.exists(calib_path):
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
            STAGE1_THRESHOLD = float(calib_data.get("optimal_accuracy_threshold", 0.55))
            
    STAGE1 = Stage1Model().to(DEVICE)
    STAGE2 = Stage2Model().to(DEVICE)
    STAGE3 = Stage3Model().to(DEVICE)
    
    STAGE1.load_state_dict(torch.load(s1_path, map_location=DEVICE))
    STAGE2.load_state_dict(torch.load(s2_path, map_location=DEVICE))
    STAGE3.load_state_dict(torch.load(s3_path, map_location=DEVICE))
    
    STAGE1.eval()
    STAGE2.eval()
    STAGE3.eval()
    
    dummy = torch.randn(1, 3, 128, 128).to(DEVICE)
    with torch.no_grad():
        o1 = STAGE1(dummy)
        p1 = (torch.softmax(o1, dim=-1)[:, 1] >= STAGE1_THRESHOLD).long()
        o2 = STAGE2(dummy, p1)
        p2 = o2.argmax(dim=-1)
        STAGE3(dummy, p2)

def predict_image(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out1 = STAGE1(img_tensor)
        prob1_nonbio = torch.softmax(out1, dim=-1)[:, 1].item()
        pred1 = torch.tensor([1 if prob1_nonbio >= STAGE1_THRESHOLD else 0], device=DEVICE)
        
        out2 = STAGE2(img_tensor, pred1)
        pred2 = out2.argmax(dim=-1)
        
        out3 = STAGE3(img_tensor, pred2)
        probs = torch.softmax(out3, dim=-1).squeeze(0).numpy()
        
    top_idx = int(probs.argmax())
    top_label = STAGE3_CLASSES[top_idx]
    
    prob_dict = {}
    for idx, cls_name in enumerate(STAGE3_CLASSES):
        prob_dict[cls_name] = float(probs[idx])
        
    return {"label": top_label, "probabilities": prob_dict}

class PredictHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                
                content_type = self.headers.get("Content-Type", "")
                image_bytes = None
                
                if "multipart/form-data" in content_type:
                    boundary = content_type.split("boundary=")[1].encode("ascii")
                    parts = body.split(b"--" + boundary)
                    for part in parts:
                        if b'name="image"' in part or b'filename=' in part:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                image_bytes = part[header_end + 4:].rstrip(b"\r\n-")
                                break
                else:
                    image_bytes = body
                    
                if not image_bytes:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "No image data provided"}).encode("utf-8"))
                    return
                    
                result = predict_image(image_bytes)
                
                response_data = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_data)))
                self.end_headers()
                self.wfile.write(response_data)
            except Exception as exc:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

def main():
    port = 5000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    load_models()
    server = ThreadingHTTPServer(("127.0.0.1", port), PredictHandler)
    print(f"Persistent inference server running on http://127.0.0.1:{port}")
    server.serve_forever()

if __name__ == "__main__":
    main()
