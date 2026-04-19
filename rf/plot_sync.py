import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import gaussian_filter1d

def sync_plot(pose_path, csi_path, output_png):
    print(f"[*] Generating Sync Visuals: {pose_path} vs {csi_path}")
    
    # Load Video Pose Data
    pose_matrix = np.load(pose_path)
    # Pose Velocity = L2 norm of the difference in XYZ coordinates per frame over all 33 joints
    pose_diffs = np.diff(pose_matrix[:, :, :3], axis=0)
    video_velocity = np.linalg.norm(pose_diffs, axis=(1, 2))
    
    # Smooth visually
    video_velocity_smooth = gaussian_filter1d(video_velocity, sigma=10)
    video_jump_frame = np.argmax(video_velocity_smooth)
    
    # Load CSI Data
    csi_data = np.load(csi_path)
    amplitudes = csi_data['amplitudes'] # [Packets, Subcarriers]
    # CSI Energy = Variance of the difference in amplitude over subcarriers
    csi_diffs = np.diff(amplitudes, axis=0)
    csi_energy = np.var(csi_diffs, axis=1)
    
    # Smooth significantly to remove 167Hz micro-jitter
    csi_energy_smooth = gaussian_filter1d(csi_energy, sigma=50)
    csi_jump_packet = np.argmax(csi_energy_smooth)
    
    print(f"[X] Suggested Video 1 Jump Frame (Index):   {video_jump_frame}")
    print(f"[X] Suggested CSI   1 Jump Packet (Index): {csi_jump_packet}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Top Plot: Video Velocity
    ax1.plot(video_velocity_smooth, color='green', linewidth=1.5)
    ax1.axvline(x=video_jump_frame, color='red', linestyle='--', label=f'Peak: Frame {video_jump_frame}')
    ax1.set_title("Video Motion Tracker (Pose Optical Velocity)")
    ax1.set_xlabel("Video Frames (30Hz System Clock)")
    ax1.set_ylabel("Geometrical Delta")
    ax1.legend()
    ax1.grid(True)
    
    # Bottom Plot: CSI Energy
    ax2.plot(csi_energy_smooth, color='purple', linewidth=1.5)
    ax2.axvline(x=csi_jump_packet, color='red', linestyle='--', label=f'Peak: Packet {csi_jump_packet}')
    ax2.set_title("Raw Environment Kinetic Energy (CSI Normalized Variance)")
    ax2.set_xlabel("Wireless Packets (~167Hz Independent Clock)")
    ax2.set_ylabel("Subcarrier RF Turbulence")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"[+] Plotted high-resolution handshake chart to {output_png}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot_sync.py <pose.npy> <dynamic_processed.npz> <output.png>")
        sys.exit(1)
        
    sync_plot(sys.argv[1], sys.argv[2], sys.argv[3])
