# train_mmf.py (adapted for CIFAR-10 classification with YOLOv1Classifier)

from csv import writer
from csv import writer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import argparse

from classification._1_dataset.dataset import CIFAR10Dataset
from model import YOLOv1ClassifierMMF, YOLOv1ClassifierMMFv1  

##################### Validate #####################
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n = len(loader)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Validation")):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits.float(), labels)

            pred = logits.argmax(dim=1)
            correct = (pred == labels).sum().item()
            total = labels.size(0)

            total_loss += loss.item()
            total_correct += correct
            total_samples += total

    avg_loss = total_loss / n
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

##################### Main #####################
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = YOLOv1ClassifierMMF(num_classes=10).to(device)

    # Load model
    best_model = torch.load(args.model_path, map_location=device)

    # model.load_state_dict(best_model['model_state_dict'])
    state_dict = best_model['model_state_dict']

    # Remove "_orig_mod.module." prefix from all keys
    cleaned_state = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("_orig_mod.module."):
            new_k = k.replace("_orig_mod.module.", "")
        elif k.startswith("module."):
            new_k = k.replace("module.", "")
        cleaned_state[new_k] = v

    model.load_state_dict(cleaned_state)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Datasets
    train_ds = CIFAR10Dataset(split='train')
    val_ds   = CIFAR10Dataset(split='test')

    # Data Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Simple classification loss: expects raw logits, internally applies softmax
    criterion = nn.CrossEntropyLoss()

    # Validate best model
    train_loss, train_acc = validate(model, train_loader, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Print final results
    print(f"Test Results for Original Model, using MMF on inference")
    print(f"Model: {args.model_path}")
    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Save final results to details.txt
    model_dir = os.path.dirname(args.model_path)  # .../runs/5/
    test_path = os.path.join(model_dir, "test_results.txt")
    with open(os.path.join(test_path), "w") as f:
        f.write(f"Test Results for Original Model, using MMF on inference\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"-------------------------\n")
        f.write(f"Test Results:\n")
        f.write(f"Train Loss: {train_loss:.6f} Acc: {train_acc:.6f}\n")
        f.write(f"Val Loss: {val_loss:.6f} Acc: {val_acc:.6f}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv1-style Classifier on CIFAR-10")

    # Model directory: model_path
    model_path = "classification/_2_train/runs_mmfv1/original_47/best_model.pth"
    parser.add_argument("--model_path", type=str, default=model_path, help="Path to checkpoint to resume from")

    # Parameters
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    main(args)