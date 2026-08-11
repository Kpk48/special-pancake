import sys
import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.hierarchy import STAGE3_CLASSES

def predict(image_path: str) -> str:
    device = torch.device("cpu")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    stage1 = Stage1Model().to(device)
    stage2 = Stage2Model().to(device)
    stage3 = Stage3Model().to(device)
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s1_path = os.path.join(root_dir, "artifacts", "hierarchical", "stage1_v2_relabeled.pt")
    s2_path = os.path.join(root_dir, "artifacts", "hierarchical", "stage2.pt")
    s3_path = os.path.join(root_dir, "artifacts", "hierarchical", "stage3.pt")
    
    stage1.load_state_dict(torch.load(s1_path, map_location=device))
    stage2.load_state_dict(torch.load(s2_path, map_location=device))
    stage3.load_state_dict(torch.load(s3_path, map_location=device))
    
    stage1.eval()
    stage2.eval()
    stage3.eval()
    
    with torch.no_grad():
        out1 = stage1(img_tensor)
        pred1 = out1.argmax(dim=-1)
        
        out2 = stage2(img_tensor, pred1)
        pred2 = out2.argmax(dim=-1)
        
        out3 = stage3(img_tensor, pred2)
        probs = torch.softmax(out3, dim=-1).squeeze(0).numpy()
        
    top_idx = int(probs.argmax())
    top_label = STAGE3_CLASSES[top_idx]
    
    prob_dict = {}
    for idx, cls_name in enumerate(STAGE3_CLASSES):
        prob_dict[cls_name] = float(probs[idx])
        
    return json.dumps({"label": top_label, "probabilities": prob_dict})

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    print(predict(sys.argv[1]))
