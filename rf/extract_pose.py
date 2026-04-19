import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO

def extract_pose(video_path, output_path):
    print(f"[*] Starting YOLOv8 Pose extraction for {video_path}")
    print("[*] Binding strictly to Hardware CUDA GPU...")
    model = YOLO('models/yolov8n-pose.pt')
    model.to('cuda') # Pin to NVIDIA GPU explicitly
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open {video_path}")
        return
        
    frames_pose = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.predict(frame, verbose=False, device='cuda')
        
        # We assume 1 person. Extract keypoints (array size: Num_Persons x 17 x 3)
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            # Getting the first person's 17 keypoints (x, y, confidence)
            frame_data = results[0].keypoints.data[0].cpu().numpy()
            # If tensor was moved to GPU or requires grad, detaching is handled by .cpu().numpy()
        else:
            # If no pose found, append zeros (17 points, 3 values each)
            frame_data = np.zeros((17, 3))
            
        frames_pose.append(frame_data)
        
        if frame_idx % 500 == 0:
            print(f"[-] Processed {frame_idx} frames...")
            
        frame_idx += 1

    cap.release()
    pose_matrix = np.array(frames_pose)
    print(f"[+] Complete. Built tensor of shape {pose_matrix.shape}")
    np.save(output_path, pose_matrix)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_pose.py <video.MOV> <output.npy>")
        sys.exit(1)
    extract_pose(sys.argv[1], sys.argv[2])
