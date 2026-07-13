# config.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Sekarang import ini akan berhasil:
from config import *

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Training
BATCH_SIZE = 32        
NUM_EPOCHS = 50
NUM_CLASSES = 7
IMG_SIZE = 48         
INPUT_SIZE = 224       

# Optimizer
LR_HEAD = 1e-3         
LR_FINETUNE = 5e-5     
UNFREEZE_EPOCH = 15    

# Classes
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]