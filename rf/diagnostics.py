import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Internal Library
from utils.csi_parser import extract_intel_csi

def do_doppler(file_path, out_path):
    print(f"[*] Processing Doppler mapping for {file_path}...")
    _, csi_complex = extract_intel_csi(file_path)
    if len(csi_complex) == 0:
        print("[!] No valid CSI sequences found.")
        return
        
    min_subc = min(100, csi_complex.shape[1])
    csi_matrix = csi_complex[:, :min_subc]
    mean_csi_time = np.mean(csi_matrix, axis=1)
    
    nperseg = min(8, len(mean_csi_time))
    nperseg = max(nperseg, 1)
    
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
        plt.colorbar(label="Magnitude")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[+] Doppler map successfully saved to {out_path}")

def do_amplitude(file_path, out_path):
    print(f"[*] Processing Amplitude mapping for {file_path}...")
    _, csi_complex = extract_intel_csi(file_path)
    if len(csi_complex) == 0:
        return
        
    min_subc = min(100, csi_complex.shape[1])
    c_vals = np.abs(csi_complex[:, :min_subc])
    
    plt.figure(figsize=(10, 6))
    for i in range(min_subc):
        plt.plot(c_vals[:, i])
    plt.title(f"CSI Amplitude - First {min_subc} Subcarriers")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Amplitude vectors strictly output to {out_path}")

def do_sync(pose_path, csi_path, out_path):
    print(f"[*] Booting Synchronizer logic with Pose:{pose_path} and CSI:{csi_path}...")
    pose_matrix = np.load(pose_path)
    pose_diffs = np.diff(pose_matrix[:, :, :3], axis=0)
    video_velocity = np.linalg.norm(pose_diffs, axis=(1, 2))
    video_velocity_smooth = gaussian_filter1d(video_velocity, sigma=10)
    video_jump_frame = np.argmax(video_velocity_smooth)
    
    csi_data = np.load(csi_path)
    csi_energy_smooth = gaussian_filter1d(np.var(np.diff(csi_data['amplitudes'], axis=0), axis=1), sigma=50)
    csi_jump_packet = np.argmax(csi_energy_smooth)
    
    print(f"[X] Suggested Video 1 Jump: Frame {video_jump_frame}")
    print(f"[X] Suggested CSI 1 Jump: Packet {csi_jump_packet}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.plot(video_velocity_smooth, color='green', linewidth=1.5)
    ax1.axvline(x=video_jump_frame, color='red', linestyle='--')
    ax1.set_title("Video Motion Tracker (Pose Optical Velocity)")
    
    ax2.plot(csi_energy_smooth, color='purple', linewidth=1.5)
    ax2.axvline(x=csi_jump_packet, color='red', linestyle='--')
    ax2.set_title("Raw Environment Kinetic Energy (CSI Normalized Variance)")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[+] Handshake output securely finalized inside {out_path}")

def do_phase(file_path, out_path):
    print(f"[*] Processing logical Phase extraction over {file_path}...")
    data = np.load(file_path)
    phases = data['phases']
    mid_subcarrier = phases.shape[1] // 2
    
    plt.figure(figsize=(12, 5))
    plt.plot(data['cumulative_times'], phases[:, mid_subcarrier], color='b', linewidth=1)
    plt.title(f"Sanitized Phase vs Time (Subcarrier {mid_subcarrier})")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[+] Saved robust Phase-Time map output onto {out_path}")

def do_peaks(file_path):
    csi_data = np.load(file_path)
    csi_energy_smooth = gaussian_filter1d(np.var(np.diff(csi_data['amplitudes'], axis=0), axis=1), sigma=50)
    peaks, properties = find_peaks(csi_energy_smooth, distance=30000, prominence=0.01)
    
    top_peaks = np.sort(peaks[np.argsort(properties['prominences'])[::-1][:2]])
    print("\n[*] Automatically Computed Exact Handshake Integers:")
    print(f"CSI Jump 1 Alignment Axis: Packet {top_peaks[0]}")
    if len(top_peaks) > 1:
        print(f"CSI Jump 2 Alignment Axis: Packet {top_peaks[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("CSI Unified Visualizer Diagnostics Tool")
    parser.add_argument('mode', choices=['amplitude', 'doppler', 'sync', 'phase', 'peaks'], help="Mode of execution.")
    parser.add_argument('--input', type=str, required=True, help="Input raw .dat file or matrix .npz file")
    parser.add_argument('--csi', type=str, help="Auxiliary parameter when syncing")
    parser.add_argument('--out', type=str, default='out.png', help="Output graph png location")
    
    args = parser.parse_args()
    
    if args.mode == 'amplitude': do_amplitude(args.input, args.out)
    elif args.mode == 'doppler': do_doppler(args.input, args.out)
    elif args.mode == 'sync': do_sync(args.input, args.csi, args.out)
    elif args.mode == 'phase': do_phase(args.input, args.out)
    elif args.mode == 'peaks': do_peaks(args.input)
