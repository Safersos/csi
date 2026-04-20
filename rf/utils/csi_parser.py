import numpy as np

def extract_intel_csi(file_path):
    """
    Parses a raw injection capture `.dat` into standard NumPy structures.
    Returns:
        hw_timestamps: Array of uint64 time metrics.
        csi_complex: Array of shape [Packets, Subcarriers] in float32 + 1j float32 complex geometry.
    """
    with open(file_path, "rb") as f:
        data = f.read()

    indices = []
    # Intel AX211 Signature Byte Locator Sequence
    for i in range(len(data) - 4):
        if data[i:i+4] == bytes.fromhex("14014d00"):
            indices.append(i)
    indices.append(len(data))
    
    if len(indices) <= 1:
        return np.array([]), np.array([])
        
    HEADER_OFFSET = 64
    
    # Establish subcarrier dimensionality to filter corrupted blocks
    sizes = [(indices[i+1] - (indices[i] + HEADER_OFFSET)) // 2 for i in range(len(indices)-1)]
    common_length = max(set(sizes), key=sizes.count)

    hw_timestamps = []
    csi_matrix = []

    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i+1]
        
        header_data = data[start:start+8]
        hw_ts = np.frombuffer(header_data, dtype=np.uint64)[0] if len(header_data) == 8 else 0
            
        payload = data[start + HEADER_OFFSET : end]
        arr = np.frombuffer(payload, dtype=np.int8)
        
        if len(arr) % 2 != 0:
            arr = arr[:-1]
            
        if len(arr) // 2 != common_length:
            continue
            
        complex_vals = arr.reshape(-1, 2)
        c_vals = complex_vals[:, 0].astype(np.float32) + 1j * complex_vals[:, 1].astype(np.float32)
        
        hw_timestamps.append(hw_ts)
        csi_matrix.append(c_vals)

    return np.array(hw_timestamps), np.array(csi_matrix)
