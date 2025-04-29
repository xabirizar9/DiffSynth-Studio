import os
import cv2
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import random
from functools import lru_cache

class VideoDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 max_frames: Optional[int] = None,
                 frame_stride: int = 1,
                 only_with_tensors: bool = False,
                 max_files: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 skip_processed: bool = True):
        """
        Args:
            root_dir (string): Directory with all the videos
            max_frames (int, optional): Maximum number of frames to load from a video
            frame_stride (int): Stride for frame sampling (1 = every frame, 2 = every other frame, etc.)
            only_with_tensors (bool): If True, only load videos that have corresponding .tensors.pth files
            max_files (int, optional): Maximum number of video files to load. If None, loads all files.
            output_dir (str, optional): Directory where processed mask files are saved. If None, uses root_dir.
            skip_processed (bool): If True, skip videos that already have mask files in output_dir.
        """
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.frame_stride = frame_stride
        self.output_dir = output_dir if output_dir is not None else root_dir
        self.skip_processed = skip_processed
        
        # Get list of video files that have corresponding .pth files
        all_files = set(os.listdir(root_dir))
        print("All files: ", len(all_files))
        
        self.video_files = []
        skipped_files = 0

        for f in all_files:
            if f.endswith(('.mp4')):
                # Check if processed mask already existsa
                mask_path = os.path.join(self.output_dir, f.replace('.mp4', '_mask.pt'))
                
                if self.skip_processed and os.path.exists(mask_path):
                    skipped_files += 1
                    continue
                
                if only_with_tensors:
                    tensor_path = os.path.join(self.root_dir, f + '.tensors.pth')
                    if os.path.exists(tensor_path):
                        self.video_files.append(f)
                else:
                    self.video_files.append(f)

        print("Found", len(self.video_files), "videos")
        print("Skipped", skipped_files, "videos")
        # Limit number of files if specified
        if max_files is not None:
            if max_files > len(self.video_files):
                print(f"Warning: max_files ({max_files}) is greater than available files ({len(self.video_files)})")
            else:
                self.video_files = self.video_files[:max_files]
                print(f"Loading {max_files} videos out of {len(self.video_files)} available")
        
        # Initialize thread pool for parallel loading
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
            
    def __len__(self) -> int:
        return len(self.video_files)
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load frames from a video file into a tensor"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Apply frame stride
            if frame_idx % self.frame_stride == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor and normalize to [0, 1]
                frame = torch.from_numpy(frame).float() / 255.0
                
                # Reorder dimensions to (C, H, W)
                frame = frame.permute(2, 0, 1)
                
                frames.append(frame)
 
                # Stop if max frames is reached
                if self.max_frames is not None and len(frames) >= self.max_frames:
                    break
                
            frame_idx += 1
            
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames loaded from video: {video_path}")
            
        # Stack frames along the first dimension
        video_tensor = torch.stack(frames, dim=0)
        
        return video_tensor
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            tuple: (video_tensor, video_filename)
                video_tensor: torch.Tensor of shape (n_frames, channels, height, width)
                video_filename: str, the filename of the video
        """
        video_file = self.video_files[idx]
        video_path = os.path.join(self.root_dir, video_file)
        
        # Load frames
        try:
            video_tensor = self._load_video_frames(video_path)
        except Exception as e:
            print(f"Error loading video {video_file}: {str(e)}")
            # Return a random video instead
            random_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(random_idx)
            
        return video_tensor, video_file
    
    def __del__(self):
        """Cleanup thread pool when dataset is destroyed"""
        self.thread_pool.shutdown()

# Example usage:
if __name__ == "__main__":
    # Create dataset with minimal processing
    dataset = VideoDataset(
        root_dir="dataset/train",
        max_frames=None,   # Load all frames
        frame_stride=1,    # Load every frame
        max_files=1000,    # Only load 1000 videos
        skip_processed=True  # Skip videos that already have mask files
    )
    
    # Get a video
    video, filename = dataset[0]
    print(f"Video shape: {video.shape}")  # Will be (n_frames, 3, height, width) 