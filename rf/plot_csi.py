import numpy as np
import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv) < 3:
    print("Usage: python plot_csi.py <input.dat> <output.png>")
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
csi_matrix = []

for i in range(len(indices) - 1):
    start = indices[i]
    end = indices[i+1]
    
    payload = data[start + HEADER_OFFSET : end]
    arr = np.frombuffer(payload, dtype=np.int8)
    
    if len(arr) % 2 != 0:
        arr = arr[:-1]
    
    if len(arr) > 0:
        complex_vals = arr.reshape(-1, 2)
        c_vals = np.sqrt(complex_vals[:, 0].astype(np.float32)**2 + complex_vals[:, 1].astype(np.float32)**2)
        csi_matrix.append(c_vals)

if not csi_matrix:
    print("No valid CSI sequences found.")
    exit(1)

min_subc = min(len(c) for c in csi_matrix)
if min_subc > 100:
    min_subc = 100

csi_matrix_np = np.array([c[:min_subc] for c in csi_matrix])

plt.figure(figsize=(10, 6))
for i in range(min_subc):
    plt.plot(csi_matrix_np[:, i])

plt.title(f"CSI Amplitude - First {min_subc} Subcarriers ({len(csi_matrix)} Packets)")
plt.xlabel("Packet/Frame Index")
plt.ylabel("CSI Amplitude")
plt.grid(True)
out_path = sys.argv[2]
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved to {out_path}")
