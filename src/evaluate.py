import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Import dari folder src
from src.dataset import get_dataloaders
from src.model import build_model
from config import *

def evaluate():
    # Gunakan CUDA jika tersedia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")
    
    # 1. Load Model
    # Pastikan build_model sesuai dengan arsitektur saat training
    model = build_model(freeze_base=False).to(device)
    model_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: File model tidak ditemukan di {model_path}")
        return

    # Load state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # 2. Get Dataloaders
    # Kita hanya butuh test_loader untuk evaluasi
    _, test_loader = get_dataloaders()
    
    all_preds = []
    all_labels = []

    print("Evaluating model...")

    # 3. Inference Pipeline
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Reporting
    # Ambil class names dari dataset
    classes = test_loader.dataset.classes
    
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 5. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    print("Evaluasi selesai!")

if __name__ == "__main__":
    evaluate()