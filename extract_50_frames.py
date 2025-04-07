import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_frames(video_path, output_folder, num_frames=50):
    """Extract evenly spaced frames from a video"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"\nProcessing: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Calculate frame indices to extract
    frame_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
    
    # Extract frames
    extracted_frames = []
    for idx in tqdm(frame_indices, desc="Extracting frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_folder, f"frame_{idx:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_frames.append(frame_filename)
    
    cap.release()
    return extracted_frames


# Input video files
before_video = "output_before.mp4"  
after_video = "output_after.mp4"   

# Output folders
before_folder = "before_frames"
after_folder = "after_frames"

# Number of frames to extract
num_frames = 50

# Process before video
print("\n=== Processing BEFORE stabilization video ===")
before_frames = extract_frames(before_video, before_folder, num_frames)

# Process after video
print("\n=== Processing AFTER stabilization video ===")
after_frames = extract_frames(after_video, after_folder, num_frames)

print("\nExtraction complete!")
print(f"Saved {len(before_frames)} frames to '{before_folder}'")
print(f"Saved {len(after_frames)} frames to '{after_folder}'")