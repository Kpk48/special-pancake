import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
import json

sys.path.insert(0, os.path.abspath("src"))

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model
from waste_classifier.hierarchical.backbone import DSConv2DBackbone
from waste_classifier.hierarchical.hierarchy import (
    STAGE1_CLASSES, STAGE2_CLASSES, STAGE3_CLASSES,
    get_stage1_label, get_stage2_label
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Check hardware specs
import platform
import psutil
print(f"OS: {platform.system()} {platform.release()}")
print(f"CPU: {platform.processor()}")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")

# Test Dataset details
raw_test = ImageFolder(root="data/final/test")
raw_train = ImageFolder(root="data/final/train")
raw_val = ImageFolder(root="data/final/val")

print(f"Train split count: {len(raw_train)}")
print(f"Val split count: {len(raw_val)}")
print(f"Test split count: {len(raw_test)}")
