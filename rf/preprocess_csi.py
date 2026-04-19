import numpy as np
import os
import sys

def process_csi_file(file_path, output_path):
    print(f"[*] Processing {file_path}...")
    with open(file_path, "rb") as f:
        data = f.read()

    indices = []
    # Find all Intel CSI signature blocks
    for i in range(len(data) - 4):
        if data[i:i+4] == bytes.fromhex("14014d00"):
            indices.append(i)
    indices.append(len(data))
    
    HEADER_OFFSET = 64
    
    hw_timestamps = []
    cumulative_times = []
    amplitudes = []
    sanitized_phases = []
    
    packet_rate = 167.0 # Average expected Hz
    
    # First, let's find the consistent subcarrier size
    sizes = []
    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i+1]
        payload = data[start + HEADER_OFFSET : end]
        length = len(payload) // 2
        sizes.append(length)
        
    if not sizes:
        print("No packets found.")
        return
        
    # Find the most common length
    common_length = max(set(sizes), key=sizes.count)
    print(f"[*] Standard subcarrier length across packets: {common_length}")

    valid_idx = 0
    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i+1]
        
        # Hardware Timestamp (First 8 bytes of the 64-byte header)
        header_data = data[start:start+8]
        if len(header_data) == 8:
            hw_ts = np.frombuffer(header_data, dtype=np.uint64)[0]
        else:
            hw_ts = 0
            
        payload = data[start + HEADER_OFFSET : end]
        arr = np.frombuffer(payload, dtype=np.int8)
        
        if len(arr) % 2 != 0:
            arr = arr[:-1]
            
        if len(arr) // 2 != common_length:
            continue # Skip malformed or truncated packets
            
        complex_vals = arr.reshape(-1, 2)
        c_vals = complex_vals[:, 0].astype(np.float32) + 1j * complex_vals[:, 1].astype(np.float32)
        
        # Math Extractions
        amp = np.abs(c_vals)
        raw_phase = np.angle(c_vals)
        
        # Phase Sanitization 
        # 1. Unwrap the phase across the subcarrier axis
        unwrapped_phase = np.unwrap(raw_phase)
        
        # 2. SFO Linear Correction (zero-center the tilt)
        subcarrier_indices = np.arange(len(unwrapped_phase))
        slope, intercept = np.polyfit(subcarrier_indices, unwrapped_phase, deg=1)
        
        # Subtract the generated line to keep 'Zero-Phase' flat
        clean_phase = unwrapped_phase - (slope * subcarrier_indices + intercept)
        
        # Append finalized lists
        hw_timestamps.append(hw_ts)
        cumulative_times.append(valid_idx / packet_rate) # Monotonic System Time Proxy
        amplitudes.append(amp)
        sanitized_phases.append(clean_phase)
        
        valid_idx += 1
        
    amplitudes = np.array(amplitudes)
    sanitized_phases = np.array(sanitized_phases)
    hw_timestamps = np.array(hw_timestamps)
    cumulative_times = np.array(cumulative_times)
    
    print(f"[*] Final Shape - Packets: {amplitudes.shape[0]}, Subcarriers: {amplitudes.shape[1]}")
    
    # Save as compressed NumPy archive
    np.savez_compressed(output_path, 
                        hw_timestamps=hw_timestamps,
                        cumulative_times=cumulative_times,
                        amplitudes=amplitudes,
                        phases=sanitized_phases)
                        
    print(f"[+] Successfully saved sanitized arrays to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_csi.py <input.dat> <output.npz>")
        sys.exit(1)
        
    process_csi_file(sys.argv[1], sys.argv[2])
