
from csv import writer
from csv import writer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import argparse

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from classification._1_dataset.dataset import CIFAR10Dataset
from classification._2_train.model import YOLOv1ClassifierMMFv7, YOLOv1Classifier


##################### Get Predictions #####################
def get_predictions_with_logits(model, loader, device):
    """Return all predictions and true labels"""
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting predictions"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.extend(outputs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_logits)

##################### Plot Top-K Accuracy Comparison #####################
def plot_topk_comparison(baseline_true, baseline_logits, mmf_true, mmf_logits, k_values):
    """Plot Top-K accuracy per class and save one plot per K"""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Convert logits to torch tensors once
    baseline_logits_t = torch.from_numpy(baseline_logits)
    mmf_logits_t = torch.from_numpy(mmf_logits)
    
    for k in k_values:
        plt.figure(figsize=(12, 7))
        x = np.arange(len(class_names))
        width = 0.35
        
        acc_base = []
        acc_mmf = []
        
        for c in range(10):
            # Baseline
            mask = (baseline_true == c)
            if mask.sum() > 0:
                class_logits = baseline_logits_t[mask]
                correct = (class_logits.topk(k, dim=1)[1] == c).any(dim=1).float().mean().item() * 100
            else:
                correct = 0
            acc_base.append(correct)
            
            # MMF
            mask = (mmf_true == c)
            if mask.sum() > 0:
                class_logits = mmf_logits_t[mask]
                correct = (class_logits.topk(k, dim=1)[1] == c).any(dim=1).float().mean().item() * 100
            else:
                correct = 0
            acc_mmf.append(correct)
        
        # Plot bars
        plt.bar(x - width/2, acc_base, width, label='Baseline', alpha=0.9, color='tab:blue')
        plt.bar(x + width/2, acc_mmf,  width, label='Best MMF',  alpha=0.9, color='tab:orange')
        
        # === Add accuracy values on top of bars ===
        for i, v in enumerate(acc_base):
            plt.text(i - width/2, v + 0.8, f"{v:.1f}%", 
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
        for i, v in enumerate(acc_mmf):
            plt.text(i + width/2, v + 0.8, f"{v:.1f}%", 
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
        # Formatting
        plt.title(f'Top-{k} Validation Accuracy per Class: Baseline vs Best MMF', fontsize=14)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel(f'Top-{k} Accuracy (%)', fontsize=12)
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = f"classification/_3_plots/runs/best/per_class_top_{k}_accuracy.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top-{k} comparison plot saved to: {save_path}")
        plt.show()


##################### Print Top-K Accuracies #####################
def print_topk_accuracies(baseline_true, baseline_logits, mmf_true, mmf_logits, k_values=[1,3,5]):
    """Print overall Top-K accuracy for both models"""
    baseline_logits_t = torch.from_numpy(baseline_logits)
    mmf_logits_t = torch.from_numpy(mmf_logits)
    
    print("\n" + "="*60)
    print("OVERALL TOP-K ACCURACY")
    print("="*60)
    
    for k in k_values:
        # Baseline
        topk_base = (baseline_logits_t.topk(k, dim=1)[1] == torch.from_numpy(baseline_true).unsqueeze(1)).any(dim=1).float().mean().item() * 100
        
        # MMF
        topk_mmf = (mmf_logits_t.topk(k, dim=1)[1] == torch.from_numpy(mmf_true).unsqueeze(1)).any(dim=1).float().mean().item() * 100
        
        print(f"Top-{k:2d} Accuracy:")
        print(f"   Baseline : {topk_base:6.2f}%")
        print(f"   Best MMF : {topk_mmf:6.2f}%")
        print(f"   Difference : {topk_mmf - topk_base:+6.2f}%\n")


##################### Main #####################
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model weights
    best_base_model = torch.load(args.base_path, map_location=device)
    best_mmf_model  = torch.load(args.mmf_path,  map_location=device)

    # 2. Define model structure
    base_model = YOLOv1Classifier(num_classes=10).to(device)
    mmf_model = YOLOv1ClassifierMMFv7(num_classes=10, weight_init_scale=0.2, quantization_levels=5).to(device)

    # model.load_state_dict(best_model['model_state_dict'])
    base_state_dict = best_base_model['model_state_dict']
    mmf_state_dict  = best_mmf_model['model_state_dict']

    # Remove "_orig_mod.module." prefix from all keys
    base_cleaned_state = {}
    for k, v in base_state_dict.items():
        new_k = k
        if k.startswith("_orig_mod.module."):
            new_k = k.replace("_orig_mod.module.", "")
        elif k.startswith("module."):
            new_k = k.replace("module.", "")
        base_cleaned_state[new_k] = v

    mmf_cleaned_state = {}
    for k, v in mmf_state_dict.items():
        new_k = k
        if k.startswith("_orig_mod.module."):
            new_k = k.replace("_orig_mod.module.", "")
        elif k.startswith("module."):
            new_k = k.replace("module.", "")
        mmf_cleaned_state[new_k] = v

    # 3. Load model weights into the model structure
    base_model.load_state_dict(base_cleaned_state)
    mmf_model.load_state_dict(mmf_cleaned_state)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        base_model = nn.DataParallel(base_model)
        mmf_model = nn.DataParallel(mmf_model)

    val_ds = CIFAR10Dataset(split='test')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate both models
    baseline_true, baseline_pred, baseline_logits = get_predictions_with_logits(base_model, val_loader, device)
    mmf_true, mmf_pred, mmf_logits = get_predictions_with_logits(mmf_model, val_loader, device)

    # Per-class top-K comparison plot
    k_values = [1, 3, 5]
    plot_topk_comparison(baseline_true, baseline_logits, mmf_true, mmf_logits, k_values)

    # === Print Overall Top-K Accuracies ===
    print_topk_accuracies(baseline_true, baseline_logits, mmf_true, mmf_logits, k_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv1-style Classifier on CIFAR-10")

    # Model directory: model_path
    base_path = "classification/_2_train/runs/best/Baseline/best_model.pth"
    mmf_path = "classification/_2_train/runs/best/Best MMF/best_model.pth"
    parser.add_argument("--base_path", type=str, default=base_path, help="Path to baseline model")
    parser.add_argument("--mmf_path", type=str, default=mmf_path, help="Path to best MMF model")


    # Parameters
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    main(args)