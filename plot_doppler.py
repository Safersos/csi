import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import sys

if len(sys.argv) < 3:
    print("Usage: python plot_doppler.py <input.dat> <output.png>")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, "rb") as f:
    data = f.read()

indices = []
for i in range(len(data) - 4):
    if data[i:i+4] == bytes.fromhex("14014d00"):
        indices.append(i)
indices.append(len(data))

HEADER_OFFSET = 64
csi_complex_history = []

for i in range(len(indices) - 1):
    start = indices[i]
    end = indices[i+1]
    
    payload = data[start + HEADER_OFFSET : end]
    arr = np.frombuffer(payload, dtype=np.int8)
    
    if len(arr) % 2 != 0:
        arr = arr[:-1]
    
    if len(arr) > 0:
        complex_vals = arr.reshape(-1, 2)
        c_vals = complex_vals[:, 0].astype(np.float32) + 1j * complex_vals[:, 1].astype(np.float32)
        csi_complex_history.append(c_vals)

if not csi_complex_history:
    print("No valid CSI sequences found.")
    exit(1)

min_subc = min(len(c) for c in csi_complex_history)
if min_subc > 100:
    min_subc = 100

csi_matrix = np.array([c[:min_subc] for c in csi_complex_history])

# Use the STFT on the mean complex CSI over subcarriers to increase SNR
mean_csi_time = np.mean(csi_matrix, axis=1)

# STFT parameters
nperseg = min(8, len(mean_csi_time))
if nperseg < 4:
    nperseg = len(mean_csi_time) if len(mean_csi_time) > 0 else 1

if len(mean_csi_time) > 1:
    f, t, Zxx = signal.stft(mean_csi_time, fs=1.0, nperseg=nperseg, noverlap=nperseg-1, return_onesided=False)
    
    from scipy.fft import fftshift
    Zxx = fftshift(Zxx, axes=0)
    f = fftshift(f)
    
    Zxx_mag = np.abs(Zxx)
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, Zxx_mag, shading='gouraud', cmap='jet')
    plt.title("Doppler-Time Map (STFT of CSI)")
    plt.ylabel("Doppler Shift (Normalized Hz)")
    plt.xlabel("Time Window (Approximated across packets)")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    out_path = sys.argv[2]
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"Generated Doppler map with {len(mean_csi_time)} packets.")
else:
    print("Not enough packets to compute STFT.")
