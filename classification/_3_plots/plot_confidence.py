
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
def get_predictions(model, loader, device):
    """Return all predictions and true labels"""
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting predictions"):
            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_confidences)

##################### Plot Confidence Comparison #####################
def plot_confidence_comparison(baseline_true, baseline_pred, baseline_conf, mmf_true, mmf_pred, mmf_conf, save_path):
    """Plot average confidence for correct vs incorrect predictions"""
    
    # Calculate average confidence
    base_correct_conf = baseline_conf[baseline_true == baseline_pred].mean()
    base_incorrect_conf = baseline_conf[baseline_true != baseline_pred].mean()
    
    mmf_correct_conf = mmf_conf[mmf_true == mmf_pred].mean()
    mmf_incorrect_conf = mmf_conf[mmf_true != mmf_pred].mean()
    
    categories = ['Correct', 'Incorrect']
    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 7))
    plt.bar(x - width/2, [base_correct_conf, base_incorrect_conf], 
            width, label='Baseline', alpha=0.9, color='tab:blue')
    plt.bar(x + width/2, [mmf_correct_conf, mmf_incorrect_conf], 
            width, label='Best MMF', alpha=0.9, color='tab:orange')

    # Add values on top
    for i, v in enumerate([base_correct_conf, base_incorrect_conf]):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for i, v in enumerate([mmf_correct_conf, mmf_incorrect_conf]):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('Average Confidence Score: Correct vs Incorrect Predictions', fontsize=13)
    plt.xlabel('Prediction Type', fontsize=12)
    plt.ylabel('Average Confidence', fontsize=12)
    plt.xticks(x, categories)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confidence comparison plot saved to: {save_path}")
    plt.show()

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
    baseline_true, baseline_pred, baseline_conf = get_predictions(base_model, val_loader, device)
    mmf_true, mmf_pred, mmf_conf = get_predictions(mmf_model, val_loader, device)

    # Confidence comparison plot
    conf_save_path = "classification/_3_plots/runs/best/confidence_correct_vs_incorrect.png"
    plot_confidence_comparison(baseline_true, baseline_pred, baseline_conf,
                              mmf_true, mmf_pred, mmf_conf, conf_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv1-style Classifier on CIFAR-10")

    # Model directory: model_path
    base_path = "classification/_2_train/runs/best/Baseline/best_model.pth"
    mmf_path = "classification/_2_train/runs/best/Best MMF/best_model.pth"
    save_path = "classification/_3_plots/runs/best/per_class_accuracy.png"
    parser.add_argument("--base_path", type=str, default=base_path, help="Path to baseline model")
    parser.add_argument("--mmf_path", type=str, default=mmf_path, help="Path to best MMF model")
    parser.add_argument("--save_path", type=str, default=save_path, help="Path to save confusion matrix plot")


    # Parameters
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    main(args)