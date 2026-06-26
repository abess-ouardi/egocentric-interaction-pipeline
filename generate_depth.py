import cv2
import torch
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pathlib import Path


# Dynamically gets the absolute directory of the current script
parent_dir = Path(__file__).resolve().parent


VIDEO_PATH = str(parent_dir / "datasets" / "push_cup.mp4")
OUTPUT_DEPTH_VIDEO = str(parent_dir / "datasets" / "depth_push_cup.mp4")


if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Could not find input video at: {VIDEO_PATH}")

# Set device: Use GPU if available, otherwise default to CPU safely
device = torch.device("cpu")

# ==========================================
# 1. LOAD DEPTH ANYTHING V2 VIA HUGGING FACE NATIVE PIPELINE
# ==========================================
print("Loading official Depth Anything V2 Small model weights...")
# FIXED: Added the required '-hf' suffix for the official transformers integration
model_id = "depth-anything/Depth-Anything-V2-Small-hf"

# Load the preprocessing image processor and the architecture model
image_processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
model.eval()

print("Depth engine initialized successfully via Transformers framework.")

# ==========================================
# 2. OPEN VIDEO AND CONFIGURE WRITER
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output video configuration
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_DEPTH_VIDEO, fourcc, fps, (width, height))

print(f"Processing {total_frames} frames to build depth profile...")

# ==========================================
# 3. PRE-COMPUTATION PROCESSING LOOP
# ==========================================
frame_idx = 0

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR (OpenCV default) to RGB 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prepare image for the model using transformers built-in processor
        inputs = image_processor(images=rgb_frame, return_tensors="pt").to(device)
        
        # Predict Depth Map
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
        
        # Interpolate/Resize depth map back to original video canvas size (1920x1080)
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Normalize depth matrix values smoothly between 0 and 255 (Grayscale conversion)
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth_normalized = depth * 0.0
            
        depth_gray = depth_normalized.cpu().numpy().astype(np.uint8)
        
        # Merge single grayscale matrix into a 3-channel layout required by VideoWriter
        depth_3channel = cv2.merge([depth_gray, depth_gray, depth_gray])
        
        # Write frame to video asset file
        out.write(depth_3channel)
        
        if frame_idx % 50 == 0:
            print(f"Pre-computed map progress: Frame {frame_idx} / {total_frames} complete.")
            
        frame_idx += 1

# Clean up memory pipelines
cap.release()
out.release()
print(f"\nSuccess! Grayscale depth profile securely compiled as: {OUTPUT_DEPTH_VIDEO}")