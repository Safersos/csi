import torch
import time
import os
import sys

# Force absolute path binding to prevent ModuleNotFoundError when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rf.dataset import get_dataloaders
from rf.model import SaferPINN
from rf.loss import PhysicsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse


def train_engine(pth_path='data/ML_Ready_Dataset.pth', epochs=150, batch_size=256, lr=2e-4, patience=10, run_name="gcn_biomechanic_v1"):
    print(f"[*] Igniting Session: {run_name} | Batch: {batch_size} | LR: {lr}")
    train_loader, val_loader = get_dataloaders(pth_path, batch_size=batch_size, w_seq=60)
    
    print("[*] Mapping Network graph directly to RTX 4050...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("[!] FATAL: CUDA hardware not bound. Safersos architecture strictly demands GPUs.")
        sys.exit(1)
        
    model = SaferPINN(w_seq=60, d_model=128, nhead=4, num_layers=3).to(device)
    
    # Optimizer & Scheduler Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    # Dynamic Plateau Breaker
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    
    criterion = PhysicsLoss(lambda_anatomy=0.5, lambda_velocity=0.1, lambda_statue=0.1).to(device)
    
    # VRAM scaling logic using explicit modern AMP mixed precision 
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"[*] Training Physics-Informed Neural Network | Batches: {len(train_loader)} | Epochs: {epochs}")
    
    best_val_loss = float('inf')
    
    # Parametric Checkpoint Routing (Prevent overwrites of past baselines)
    run_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"models/{run_name}.pt")
    if os.path.exists(run_ckpt):
        print(f"[*] Warning: Re-running over existing baseline trace '{run_name}.pt'...")
    
    try:
        for epoch in range(1, epochs + 1):
            
            # The GAT structurally enforces topology organically.
            dyn_anatomy = 0.2 + (0.8 * (epoch / epochs)) # Scales mildly from 0.2 to 1.0
            criterion.lambda_anatomy = dyn_anatomy
            
            model.train()
            epoch_loss = 0.0
            start_time = time.time()
            
            for batch_idx, (x, y_root, y_local) in enumerate(train_loader):
                
                x = x.to(device)
                y_root = y_root.to(device)
                y_local = y_local.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred_root, pred_local = model(x)
                    loss, L_mse, L_vel, L_anat, L_stat = criterion(pred_root, pred_local, y_root, y_local, x)
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                
                if batch_idx % 25 == 0 and batch_idx > 0:
                    print(f"    [Epoch {epoch} | Batch {batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} (MSE: {L_mse.item():.2f} | Anat: {L_anat.item():.2f} @ x{dyn_anatomy:.2f})")
            
            # Validation Loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                 for x_v, y_root_v, y_local_v in val_loader:
                     x_v = x_v.to(device)
                     y_root_v = y_root_v.to(device)
                     y_local_v = y_local_v.to(device)
                     with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                         pred_r_v, pred_l_v = model(x_v)
                         v_loss, _, _, _, _ = criterion(pred_r_v, pred_l_v, y_root_v, y_local_v, x_v)
                     val_loss += v_loss.item()
                     
            avg_train_loss = epoch_loss/len(train_loader)
            avg_val_loss = val_loss/len(val_loader)
            
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"[*] Plateau Breaker engaged! Learning Rate decayed to {new_lr}")
                     
            print(f"[*] Epoch {epoch} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time()-start_time:.2f}s")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"[*] New Target Reached! Securing isolated trace checkpoint '{run_name}.pt'...")
                torch.save(model.state_dict(), run_ckpt)
                
    except KeyboardInterrupt:
        print("\n[!] User aborted training run. Finalizing logs prematurely...")
        
    # Write to Experiment Logbook
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_logs.md")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
             f.write("# Safersos Core Training Logbook\n\n| Date               | Run Key           | Epochs | Batch | LR   | Best Val Loss |\n|--------------------|-------------------|--------|-------|------|---------------|\n")
             
    with open(log_path, "a") as f:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"| {ts} | `{run_name}` | {epochs} | {batch_size} | {lr} | {best_val_loss:.4f} |\n")
        print(f"\n[*] Experiment fully logged to train_logs.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SaferPINN Training Engine")
    parser.add_argument('-e', '--epochs', type=int, default=150, help='Total epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='Batch size constraints')
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial AdamW Learning Rate')
    parser.add_argument('-p', '--patience', type=int, default=10, help='Epochs to wait before slashing LR on plateau')
    parser.add_argument('-n', '--run_name', type=str, default="gcn_biomechanic_v1", help='Experiment tracker tag')
    args = parser.parse_args()
    
    train_engine(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience, run_name=args.run_name)
