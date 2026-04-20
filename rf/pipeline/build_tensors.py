import numpy as np
import sys
import torch
from scipy.interpolate import PchipInterpolator

def build_tensors(csi_path, static_path, pose1_path, pose2_path, out_path, v1_jump, csi1_jump, v2_jump, csi2_jump):
    print(f"[*] Loading Data...")
    
    # Load Master Vectors
    dyn_data = np.load(csi_path)
    dyn_times = dyn_data['cumulative_times']
    dyn_amp = dyn_data['amplitudes']
    dyn_phase = dyn_data['phases']
    
    stat_data = np.load(static_path)
    stat_amp = stat_data['amplitudes']
    stat_phase = stat_data['phases']
    
    pose1 = np.load(pose1_path)
    pose2 = np.load(pose2_path)
    
    print("[*] Synchronizing Clocks and Excising the Hardware Gap...")
    # Calculate Anchor Timestamps
    t_csi1 = dyn_times[csi1_jump]
    t_csi2 = dyn_times[csi2_jump]
    
    # Video 1 Bounds
    t_start1 = t_csi1 - (v1_jump / 30.0)
    t_end1 = t_csi1 + ((len(pose1) - 1 - v1_jump) / 30.0)
    
    # Video 2 Bounds
    t_start2 = t_csi2 - (v2_jump / 30.0)
    t_end2 = t_csi2 + ((len(pose2) - 1 - v2_jump) / 30.0)
    
    # Find active packets representing only recorded action
    mask_v1 = (dyn_times >= t_start1) & (dyn_times <= t_end1)
    mask_v2 = (dyn_times >= t_start2) & (dyn_times <= t_end2)
    
    csi_idx_v1 = np.where(mask_v1)[0]
    csi_idx_v2 = np.where(mask_v2)[0]
    
    print(f"[-] Video 1 Matched Packets: {len(csi_idx_v1)}")
    print(f"[-] Video 2 Matched Packets: {len(csi_idx_v2)}")
    
    # Create the upscaling interpolators (30Hz -> Continuous PCHIP Function)
    print("[*] Executing PCHIP Interpolation into 167Hz Tensor Resolution...")
    
    t_pose1 = t_csi1 + (np.arange(len(pose1)) - v1_jump) / 30.0
    # Flatten spatial dims to interpolate multiple channels
    pose1_flat = pose1.reshape(len(pose1), -1) 
    interp1 = PchipInterpolator(t_pose1, pose1_flat, axis=0)
    y1_interp = interp1(dyn_times[csi_idx_v1]).reshape(len(csi_idx_v1), 17, 3)
    
    t_pose2 = t_csi2 + (np.arange(len(pose2)) - v2_jump) / 30.0
    pose2_flat = pose2.reshape(len(pose2), -1)
    interp2 = PchipInterpolator(t_pose2, pose2_flat, axis=0)
    y2_interp = interp2(dyn_times[csi_idx_v2]).reshape(len(csi_idx_v2), 17, 3)
    
    # Stitch valid unified arrays
    valid_csi_idx = np.concatenate([csi_idx_v1, csi_idx_v2])
    unified_posed = np.concatenate([y1_interp, y2_interp], axis=0)
    
    unified_amp = dyn_amp[valid_csi_idx]
    unified_phase = dyn_phase[valid_csi_idx]
    
    # Z-Score Background Erasure (Blinding the walls)
    print("[*] Erasing the ambient space signature (Z-Score Normalization)...")
    mu_amp = np.mean(stat_amp, axis=0)
    sig_amp = np.std(stat_amp, axis=0)
    sig_amp[sig_amp == 0] = 1e-6 # prevent div by zero
    z_amp = (unified_amp - mu_amp) / sig_amp
    
    mu_phase = np.mean(stat_phase, axis=0)
    sig_phase = np.std(stat_phase, axis=0)
    sig_phase[sig_phase == 0] = 1e-6
    z_phase = (unified_phase - mu_phase) / sig_phase
    
    # Shape Tensor: [Total_Packets, Subcarriers, 2]
    csi_tensor = np.stack([z_amp, z_phase], axis=-1)
    
    print(f"[+] Final Export Shape: CSI {csi_tensor.shape} | POSE {unified_posed.shape}")
    
    torch.save({
        'csi_x': torch.tensor(csi_tensor, dtype=torch.float32),
        'pose_y': torch.tensor(unified_posed, dtype=torch.float32)
    }, out_path)
    print(f"[SUCCESS] High-Density Dataset formally published to {out_path}!")

if __name__ == "__main__":
    if len(sys.argv) < 10:
        print("Usage: python build_tensors.py <dynamic_csi.npz> <static_csi.npz> <pose1> <pose2> <out> <v1_idx> <csi1_idx> <v2_idx> <csi2_idx>")
        sys.exit(1)
        
    build_tensors(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                  int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]))
