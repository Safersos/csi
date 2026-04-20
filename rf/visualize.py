import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import argparse

# Change working directory so it can find things
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from rf.model import SaferPINN
from rf.dataset import CSIPoseDataset

# YOLOv8 Skeleton Bone Mapping (17 Keypoints)
BONES = [
    (0, 1), (1, 3),      # Left face
    (0, 2), (2, 4),      # Right face
    (5, 6),              # Shoulders
    (5, 7), (7, 9),      # Left arm
    (6, 8), (8, 10),     # Right arm
    (5, 11), (6, 12),    # Torso
    (11, 12),            # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)   # Right leg
]

def draw_skeleton(ax, points, title, color="blue"):
    """
    points: [17, 3] representing x, y, conf
    """
    ax.set_title(title, color='white')
    ax.set_facecolor('black')
    
    # Extract x and y coordinates (ignore confidence for drawing lines)
    x = points[:, 0]
    y = points[:, 1]
    
    # Invert Y axis so head is at top (pixel coordinates start top-left)
    ax.invert_yaxis()
    
    # Plot joints
    ax.scatter(x, y, c='red', s=40, zorder=5)
    
    # Draw bones
    for p1, p2 in BONES:
        ax.plot([x[p1], x[p2]], [y[p1], y[p2]], c=color, linewidth=2, zorder=3)
        
    ax.set_aspect('equal')
    ax.axis('off')

def main(ckpt_name):
    print(f"[*] Initiating Phase 4: Hardware Inference Visualizer")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Targeting Device: {device}")
    
    # 1. Load Dataset
    data_path = os.path.join(project_dir, 'data', 'ML_Ready_Dataset.pth')
    if not os.path.exists(data_path):
        print(f"[!] FATAL: Dataset missing at {data_path}")
        return
        
    full_dataset = CSIPoseDataset(data_path, w_seq=60)
    
    # Re-split to extract Validation exactly like training did
    # We use a fixed seed if we want identical split, but Pytorch random_split uses local generator
    torch.manual_seed(42)  # align with train.py
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    print(f"[*] Loaded Validation Dataset: {len(val_ds)} physical temporal chunks.")
    
    # 2. Architect Model & Load Weights
    model = SaferPINN(w_seq=60, d_model=128, nhead=4, num_layers=3).to(device)
    
    model_path = os.path.join(project_dir, 'models', f"{ckpt_name}.pt")
    if not os.path.exists(model_path):
        print(f"[!] FATAL: Checkpoint {ckpt_name}.pt not found!")
        return
        
    print(f"[*] Loading Physical Model Matrix: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 3. Randomize selection
    num_samples = 3
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 10))
    fig.patch.set_facecolor('#1e1e1e')
    fig.suptitle(f"Ground Truth (YOLOv8)  vs  Predicted ({ckpt_name})", color='white', fontsize=16)
    
    indices = random.sample(range(len(val_ds)), num_samples)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            x_csi, y_root_true, y_local_true = val_ds[idx]
            
            # Add batch dim and send to device
            x_csi = x_csi.unsqueeze(0).to(device)
            
            # Predict
            pred_root, pred_local = model(x_csi)  
            
            # Reconstruct Predicted Absolute Geometry
            pred_root_last = pred_root[0, -1].cpu() # [2]
            pred_local_last = pred_local[0, -1].cpu() # [17, 2]
            y_pred = pred_local_last + pred_root_last.unsqueeze(0) # [17, 2]
            y_pred = y_pred.numpy()
            
            # Reconstruct True Absolute Geometry
            y_root_last = y_root_true[-1] # [2]
            y_local_last = y_local_true[-1] # [17, 2]
            y_true = y_local_last + y_root_last.unsqueeze(0) # [17, 2]
            y_true = y_true.numpy()
            
            # Row 0: True
            draw_skeleton(axes[0, i], y_true, f"Validation Target {idx}", color="#00ff00")
            
            # Row 1: Pred
            draw_skeleton(axes[1, i], y_pred, f"Model Wi-Fi Prediction", color="#00c3ff")

    # Save Output
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(project_dir, f'artifacts_inference_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor='#1e1e1e')
    print(f"\n[*] SUCCESS: Matrix rendered to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SaferPINN Visualizer")
    parser.add_argument('-m', '--model', type=str, default="v1.0_DualHead_Spaghetti", help='Checkpoint name (excluding .pt extension)')
    args = parser.parse_args()
    main(args.model)
