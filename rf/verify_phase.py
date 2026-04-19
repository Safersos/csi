import numpy as np
import matplotlib.pyplot as plt
import sys

def verify_phase(file_path, output_png):
    data = np.load(file_path)
    phases = data['phases']
    times = data['cumulative_times']
    
    # Pick a middle subcarrier (typically has good SNR)
    mid_subcarrier = phases.shape[1] // 2
    
    phase_timeline = phases[:, mid_subcarrier]
    
    plt.figure(figsize=(12, 5))
    plt.plot(times, phase_timeline, color='b', linewidth=1)
    plt.title(f"Sanitized Phase vs Time (Subcarrier {mid_subcarrier})")
    plt.xlabel("Cumulative Time Proxy (seconds)")
    plt.ylabel("Radians (Zero-Centered)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"[+] Saved verification plot to {output_png}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_phase.py <input.npz> <output.png>")
        sys.exit(1)
        
    verify_phase(sys.argv[1], sys.argv[2])
