import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.dataset import get_dataloaders
from src.model import build_model, unfreeze_top_layers
from config import *

CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "checkpoint.pth")


def save_checkpoint(epoch, model, optimizer, scheduler, best_acc, phase):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'phase': phase,  # simpan phase 1 atau 2
    }, CHECKPOINT_PATH)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, scheduler):
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Skip load optimizer kalau size mismatch
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    except ValueError:
        print("Optimizer mismatch, reset optimizer & scheduler")

    print(f"Resumed from epoch {checkpoint['epoch']} | Phase: {checkpoint['phase']}")
    return checkpoint['epoch'], checkpoint['best_acc'], checkpoint['phase']


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0

    def check(self, val_acc):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            return False  # jangan stop
        else:
            self.counter += 1
            print(f"  ⏳ Early stopping counter: {self.counter}/{self.patience}")
            return self.counter >= self.patience  # stop kalau udah patience


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    train_loader, test_loader = get_dataloaders()
    model = build_model(freeze_base=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc = 0.0
    start_epoch = 1
    phase = 1

    # Auto-detect checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        ans = input("Checkpoint ditemukan! Resume? (y/n): ")
        if ans.lower() == 'y':
            start_epoch, best_acc, phase = load_checkpoint(model, optimizer, scheduler)
            start_epoch += 1  # lanjut epoch berikutnya

            # Kalau checkpoint di phase 2, langsung unfreeze    
            if phase == 2:
                model = unfreeze_top_layers(model, n_layers=20)
                print(">>> Restored to fine-tuning phase")

    early_stopping = EarlyStopping(patience=5)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        # Switch ke phase 2
        if epoch == UNFREEZE_EPOCH and phase == 1:
            phase = 2
            model = unfreeze_top_layers(model, n_layers=20)
            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_FINETUNE
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - epoch)
            print(">>> Switched to fine-tuning phase")

        # --- Training ---
        model.train()
        train_loss, train_correct = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # --- Validation ---
        model.eval()
        val_loss, val_correct = 0, 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(test_loader.dataset)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/len(test_loader):.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step()

        # Simpan checkpoint tiap epoch
        save_checkpoint(epoch, model, optimizer, scheduler, best_acc, phase)

        # Simpan best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_model.pth"))
            print(f"Best model saved, Val Acc: {val_acc:.4f}")

        if early_stopping.check(val_acc):
            print("Early stopping triggered! Training selesai.")
            break


if __name__ == "__main__":
    train() 