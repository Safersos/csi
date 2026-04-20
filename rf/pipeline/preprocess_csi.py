import numpy as np
import os
import sys

# Ensure absolute module pathing directly resolves to rf/ for structural libraries
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.csi_parser import extract_intel_csi

def process_csi_file(file_path, output_path):
    print(f"[*] Extracting Array Dimensions & Initializing Phase Sanitation over {file_path}...")
    
    hw_timestamps, csi_complex = extract_intel_csi(file_path)
    if len(csi_complex) == 0:
        print("[!] Execution Failure: Physical hardware telemetry arrays empty or corrupted.")
        return
        
    packet_rate = 167.0 # Target metric
    cumulative_times = np.arange(len(hw_timestamps)) / packet_rate
    
    # 1. Global Amplitude Map
    amplitudes = np.abs(csi_complex)
    
    # 2. Advanced Phase Filtering & Carrier Calibration
    sanitized_phases = np.zeros_like(amplitudes)
    subcarrier_indices = np.arange(amplitudes.shape[1])
    
    print("[*] Processing advanced SFO Slope Normalization loops...")
    for i in range(len(csi_complex)):
        raw_phase = np.angle(csi_complex[i])
        unwrapped_phase = np.unwrap(raw_phase)
        
        # SFO Linear Tilt Excision
        slope, intercept = np.polyfit(subcarrier_indices, unwrapped_phase, deg=1)
        clean_phase = unwrapped_phase - (slope * subcarrier_indices + intercept)
        sanitized_phases[i] = clean_phase
        
    print(f"[*] Post-Sanitization Tensor Map Output - Packets: {amplitudes.shape[0]}, Subcarriers: {amplitudes.shape[1]}")
    
    np.savez_compressed(output_path, 
                        hw_timestamps=hw_timestamps,
                        cumulative_times=cumulative_times,
                        amplitudes=amplitudes,
                        phases=sanitized_phases)
                        
    print(f"[+] Successfully committed flawless tensor output physically onto {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_csi.py <input.dat> <output.npz>")
        sys.exit(1)
        
    process_csi_file(sys.argv[1], sys.argv[2])
