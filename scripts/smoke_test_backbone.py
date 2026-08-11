import sys
sys.path.insert(0, "src")
from waste_classifier.hierarchical.backbone import MobileNetV3Backbone
from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
import torch

b = MobileNetV3Backbone()
x = torch.randn(2, 3, 224, 224)
out = b(x)
print("Backbone output shape:", out.shape, "| dtype:", out.dtype)
assert out.shape == (2, 128), f"Expected (2,128) got {out.shape}"

s1 = Stage1Model()
s2 = Stage2Model()
s3 = Stage3Model(num_classes=8)
cond1 = torch.zeros(2, dtype=torch.long)
cond2 = torch.zeros(2, dtype=torch.long)
o1 = s1(x)
o2 = s2(x, cond1)
o3 = s3(x, cond2)
print("Stage1 out:", o1.shape)
print("Stage2 out:", o2.shape)
print("Stage3 out:", o3.shape)
print("ALL OK")
