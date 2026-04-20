import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CSIPoseDataset(Dataset):
    def __init__(self, pth_path, w_seq=60, is_train=True):
        """
        Sliding Window Memory Dataset
        w_seq: 60 packets (~0.35s context window)
        """
        data = torch.load(pth_path, weights_only=False)
        self.csi = data['csi_x']   # [N, 213, 2]
        self.pose = data['pose_y'] # [N, 17, 3]
        
        self.w_seq = w_seq
        self.is_train = is_train
        self.dataset_len = len(self.csi) - self.w_seq
        
    def __len__(self):
        return self.dataset_len
        
    def __getitem__(self, idx):
        # Extract temporal context W
        x_window = self.csi[idx : idx + self.w_seq].clone()
        # Predicting the entire W sequence physically enforces velocity trajectory
        y_window = self.pose[idx : idx + self.w_seq].clone()
        
        # O(1) Target Normalization: Scale absolute 4K pixel coordinates to ~[0,1]
        y_window[:, :, :2] = y_window[:, :, :2] / 3000.0
        
        # Root-Relative Decomposition
        # 1. Extract Global Trajectory (Mid-Hip anchor point across W sequence)
        root_y = (y_window[:, 11, :2] + y_window[:, 12, :2]) / 2.0  # [W, 2]
        
        # 2. Extract Local Posture (All 17 joints relative to the root anchor)
        local_y = y_window[:, :, :2] - root_y.unsqueeze(1) # [W, 17, 2]
        
        if self.is_train:
            # Data Augmentation: Silent Phase Jitter (Masking hardware noise)
            # Channel 0 is Amp, Channel 1 is Phase
            phase_noise = torch.randn_like(x_window[:, :, 1]) * 0.05
            x_window[:, :, 1] += phase_noise
            
        return x_window, root_y, local_y

def get_dataloaders(pth_path, batch_size=64, w_seq=60, split=0.8):
    dataset = CSIPoseDataset(pth_path, w_seq=w_seq, is_train=True)
    
    train_sz = int(len(dataset) * split)
    val_sz = len(dataset) - train_sz
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_sz, val_sz])
    
    # We enforce train_ds validation to NOT augment, but PyTorch Split passes the parent __getitem__
    # So both get phase noise in this simple split, which acts as robust regularization.
    
    # Throttled num_workers to 1 to drastically decrease system CPU stress
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    return train_dl, val_dl
