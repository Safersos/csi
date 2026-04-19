import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Load CSI Data
csi_data = np.load('data/dynamic_room_processed.npz')
amplitudes = csi_data['amplitudes']

# Energy extraction
csi_diffs = np.diff(amplitudes, axis=0)
csi_energy = np.var(csi_diffs, axis=1)
csi_energy_smooth = gaussian_filter1d(csi_energy, sigma=50)

# We want the 2 most prominent peaks separated by at least roughly the gap duration
# 167Hz packet rate. A 3-minute video = 30000 packets. Distance should be ~30,000.
peaks, properties = find_peaks(csi_energy_smooth, distance=30000, prominence=0.01)

# Sort by prominence to get top 2 peaks
top_indices = np.argsort(properties['prominences'])[::-1][:2]
top_peaks = peaks[top_indices]
top_peaks = np.sort(top_peaks) # chronological

print("[*] Automatically Computed Exact Handshake Integers:")
print(f"CSI Jump 1 (Video 4135 Match): Packet {top_peaks[0]}")
if len(top_peaks) > 1:
    print(f"CSI Jump 2 (Video 4136 Match): Packet {top_peaks[1]}")
