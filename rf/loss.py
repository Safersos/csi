import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsLoss(nn.Module):
    def __init__(self, lambda_anatomy=0.1, lambda_velocity=0.05, lambda_statue=0.1):
        super().__init__()
        self.lambda_anatomy = lambda_anatomy
        self.lambda_velocity = lambda_velocity
        self.lambda_statue = lambda_statue
        
        # 17 Keypoints. Higher weights for Torso/Core elements to satisfy core target dynamics.
        base_weights = torch.ones(17)
        # Emphasize center of mass tracking heavily
        base_weights[[5, 6, 11, 12]] = 2.0  
        # De-emphasize jittery extremeties (Wrists, Ankles, Ears)
        base_weights[[9, 10, 15, 16, 3, 4]] = 0.5 
        
        # Dims: [1, 1, 17, 1] for broadcasting over [B, W, 17, 2]
        self.register_buffer('joint_weights', base_weights.view(1, 1, 17, 1))

        # Absolute Physiological Bone Mean Baselines normalized to the 3000.0 ratio
        self.biological_bones = {
            (5, 7): 82.60 / 3000.0,   
            (6, 8): 80.08 / 3000.0,   
            (7, 9): 54.78 / 3000.0,   
            (8, 10): 70.86 / 3000.0,  
            (5, 6): 83.62 / 3000.0,   
            (11, 12): 54.93 / 3000.0, 
            (5, 11): 137.71 / 3000.0, 
            (6, 12): 143.54 / 3000.0, 
            (11, 13): 95.82 / 3000.0, 
            (12, 14): 97.92 / 3000.0, 
            (13, 15): 119.04 / 3000.0,
            (14, 16): 122.34 / 3000.0 
        }

    def forward(self, pred_root, pred_local, target_root, target_local, x_csi):
        # 1. Global Trajectory Loss (Navigator)
        loss_root = F.mse_loss(pred_root, target_root)
        
        # 2. Local Posture Loss (Biomechanic)
        loss_local = F.mse_loss(pred_local * self.joint_weights, target_local * self.joint_weights)
        
        # 3. Velocity Smoothing on Global Trajectory
        pred_vel = pred_root[:, 1:, :] - pred_root[:, :-1, :] 
        targ_vel = target_root[:, 1:, :] - target_root[:, :-1, :]
        vel_loss = F.mse_loss(pred_vel, targ_vel)
        
        # 4. Anti-Statue Guardrail (Velocity Consistency)
        # If CSI dynamic variance is high (turbulence), penalize zero physical velocity
        # x_csi: [B, W, 213, 2] -> 0 is Amp, 1 is Phase
        # Calculate Phase variance per frame
        var_csi = torch.var(x_csi[:, 1:, :, 1], dim=2) # [B, W-1]
        var_threshold = 0.8
        var_mask = (var_csi > var_threshold).float()
        
        # Epsilon inserted natively into the squared sum to prevent infinite gradient divergence at exact zero
        vel_magnitude = torch.sqrt(torch.sum(pred_vel**2, dim=-1) + 1e-8) # [B, W-1]
        
        # If mask is 1 (high variance) and velocity is small -> high penalty
        anti_statue_loss = torch.mean(var_mask * torch.exp(-50.0 * vel_magnitude))
        
        # 5. Anatomy Bone Length Distension Calculation
        l_anatomy = 0.0
        for (j1, j2), true_len in self.biological_bones.items():
            # Relative offsets subtract out perfectly: (P1 - Root) - (P2 - Root) = P1 - P2
            bone_vectors = pred_local[:, :, j1, :] - pred_local[:, :, j2, :]
            # Protected numerical magnitude calculation for AMP limits
            bone_lengths = torch.sqrt(torch.sum(bone_vectors**2, dim=-1) + 1e-8)
            bone_dev = F.smooth_l1_loss(bone_lengths, torch.full_like(bone_lengths, true_len))
            l_anatomy += bone_dev
            
        l_anatomy = l_anatomy / len(self.biological_bones)

        total_loss = loss_root + loss_local + (self.lambda_velocity * vel_loss) + (self.lambda_statue * anti_statue_loss) + (self.lambda_anatomy * l_anatomy)
        
        mse_metric = loss_root + loss_local
        
        return total_loss, mse_metric, vel_loss, l_anatomy, anti_statue_loss
