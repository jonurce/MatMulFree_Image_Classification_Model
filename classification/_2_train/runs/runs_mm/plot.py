from tbparse import SummaryReader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === CONFIG ===

# Experiment name (CHANGE)
runs_folder = "runs/runs_mmfv2"

# Root folder containing all experiment folders
log_dir = f"/home/jon/Workspace/Low_Power_Satellite_6DoF_Pose_Estimation/classification/_2_train/{runs_folder}"          

# or "Val/epoch/accuracy", "Train/epoch/accuracy", etc
metric_name = "Val/epoch/accuracy"

# For plot titles and filenames
metric_plot_name = "MMF Model Hyperparameter Tuning: Validation Accuracy"

if runs_folder.endswith('mm'):
    metric_plot_name = "Baseline Model Hyperparameter Tuning: Validation Accuracy"

elif runs_folder.endswith('mmf'):
    metric_plot_name = "MMF Model Hyperparameter Tuning: Validation Accuracy"

elif runs_folder.endswith('mmfv2'):
    metric_plot_name = "MMF Model Modifying Number of Channels: Validation Accuracy"

elif runs_folder.endswith('mmfv5'):
    metric_plot_name = "MMF Model Modifying Weight Initialization Scale: Validation Accuracy"

elif runs_folder.endswith('mmfv7'):
    metric_plot_name = "MMF Model Ternary, Quinary, Septenary, Nonary: Validation Accuracy"




# Find all experiment runs
runs = list(Path(log_dir).glob("**/*tfevents*"))  # or better structure

print(f"Found {len(runs)} event files")

# Read all runs
reader = SummaryReader(log_dir, extra_columns={'dir_name'})
df = reader.scalars

# Filter only the metric you care about
df_metric = df[df['tag'] == metric_name].copy()

# 1. Get final accuracy for each run
final_acc = df_metric.groupby('dir_name').last()[['step', 'value']].reset_index()
final_acc = final_acc.rename(columns={'value': 'final_accuracy'})
final_acc = final_acc.sort_values('final_accuracy', ascending=False)

# Save results
save_csv = f"/home/jon/Workspace/Low_Power_Satellite_6DoF_Pose_Estimation/classification/_3_plots/{runs_folder}/{metric_plot_name}.csv"          
final_acc.to_csv(save_csv, index=False)
print(f"CSV saved successfully")

# 2. Plot training curves
plt.figure(figsize=(12, 12))
unique_runs = sorted(df_metric['dir_name'].unique(),  key=lambda x: int(''.join(filter(str.isdigit, str(x))) or 0))
sns.lineplot(data=df_metric, x='step', y='value', hue='dir_name', hue_order=unique_runs,linewidth=1.8,alpha=0.85)
plt.title(metric_plot_name, fontsize=14)
plt.xlabel('Steps / Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title='Run ID', title_fontsize=11,fontsize=9,bbox_to_anchor=(1.02, 1), loc='upper left',frameon=True)
plt.tight_layout()

# Save plot
plot_path = f"/home/jon/Workspace/Low_Power_Satellite_6DoF_Pose_Estimation/classification/_3_plots/{runs_folder}/{metric_plot_name}.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

print(f"Plot saved successfully")

