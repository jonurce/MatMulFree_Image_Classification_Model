
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


################## Plott Weight Distribution ##################
def plot_weight_distribution(base_model, mmf_model, save_path):
    """Plot weight distribution: Baseline vs MMF (raw + quantized)"""
    base_weights = []
    mmf_raw_weights = []
    mmf_quant_weights = []
    
    # Collect Baseline weights
    for name, param in base_model.named_parameters():
        if 'weight' in name and param.requires_grad:
            base_weights.append(param.data.cpu().numpy().flatten())
    
    # Collect MMF raw weights
    for name, param in mmf_model.named_parameters():
        if 'weight' in name and param.requires_grad:
            w = param.data.cpu()
            mmf_raw_weights.append(w.numpy().flatten())
            
    mmf_raw = np.concatenate(mmf_raw_weights)
    s_w = 1.0 / np.abs(mmf_raw).mean().clip(min=1e-8)

    # Collect MMF quantizedweights
    for name, param in mmf_model.named_parameters():
        if 'weight' in name and param.requires_grad:
            w = param.data.cpu()
            w_quant = (s_w * w).round().clamp(-2, 2) / s_w
            mmf_quant_weights.append(w_quant.numpy().flatten())
    
    base_all = np.concatenate(base_weights)
    mmf_raw = np.concatenate(mmf_raw_weights)
    mmf_quant = np.concatenate(mmf_quant_weights)
    
    plt.figure(figsize=(13, 8))
    
    plt.hist(base_all, bins=200, alpha=0.7, label='Baseline (Standard)', 
             color='tab:blue', density=False, linewidth=0.5)
    plt.hist(mmf_raw, bins=200, alpha=0.7, label='MMF (Raw Float32)', 
             color='tab:orange', density=False, linewidth=0.5)
    plt.hist(mmf_quant, bins=200, alpha=0.85, label='MMF (Quantized)', 
             color='tab:red', density=False, linewidth=0.5)
    
    plt.title('Weight Distribution: Baseline vs MMF (Raw vs Quantized)', fontsize=14)
    plt.xlabel('Weight Value', fontsize=12)
    plt.ylabel('Number of Weights', fontsize=12)
    # plt.yscale('log')
    plt.xlim(-1, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Weight distribution plot saved to: {save_path}")
    plt.show()
    
    # Statistics
    print(f"\nWeight Statistics:")
    print(f"Baseline - Mean: {base_all.mean():.4f}, Std: {base_all.std():.4f}")
    print(f"MMF Raw   - Mean: {mmf_raw.mean():.4f}, Std: {mmf_raw.std():.4f}")
    print(f"MMF Quant - Mean: {mmf_quant.mean():.4f}, Std: {mmf_quant.std():.4f}")
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

    # 4. Plot weight distribution
    plot_weight_distribution(base_model, mmf_model, args.save_path)

    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv1-style Classifier on CIFAR-10")

    # Model directory: model_path
    base_path = "classification/_2_train/runs/best/Baseline/best_model.pth"
    mmf_path = "classification/_2_train/runs/best/Best MMF/best_model.pth"
    save_path = "classification/_3_plots/runs/best/weight_distribution.png"
    parser.add_argument("--base_path", type=str, default=base_path, help="Path to baseline model")
    parser.add_argument("--mmf_path", type=str, default=mmf_path, help="Path to best MMF model")
    parser.add_argument("--save_path", type=str, default=save_path, help="Path to save confusion matrix plot")


    # Parameters
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    main(args)