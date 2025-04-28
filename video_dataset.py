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
                 height: int,
                 width: int,
                 max_frames: Optional[int] = None,
                 frame_stride: int = 1,
                 max_files: Optional[int] = None):
        """
        Args:
            root_dir (string): Directory with all the videos
            target_size (tuple, optional): Target size for resizing frames (height, width). If None, keeps original size.
            max_frames (int, optional): Maximum number of frames to load per video. If None, loads all frames.
            frame_stride (int): Stride for frame sampling (1 = every frame, 2 = every other frame, etc.)
            max_files (int, optional): Maximum number of video files to load. If None, loads all files.
        """
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.max_frames = max_frames
        self.frame_stride = frame_stride
        
        # Get list of video files that have corresponding .pth files
        all_files = set(os.listdir(root_dir))
        print("All files: ", len(all_files))
        self.video_files = []

        for f in all_files:
            if f.endswith(('.mp4')):
                if os.path.exists(self.root_dir + '/' + f + '.tensors.pth'):
                    self.video_files.append(f)
        print("Loading ", len(self.video_files), " videos")
        # Limit number of files if specified
        if max_files is not None:
            if max_files > len(self.video_files):
                print(f"Warning: max_files ({max_files}) is greater than available files ({len(self.video_files)})")
            else:
                self.video_files = self.video_files[:max_files]
                print(f"Loading {max_files} videos out of {len(self.video_files)} available")
        
        # Initialize thread pool for parallel loading
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Store video metadata
        self.video_metadata = []
        for video_file in self.video_files:
            cap = cv2.VideoCapture(os.path.join(root_dir, video_file))
            if not cap.isOpened():
                continue
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.video_metadata.append({
                'file': video_file,
                'frame_count': frame_count,
                'fps': fps
            })
            cap.release()
            
    def __len__(self) -> int:
        return len(self.video_files)
    
    @lru_cache(maxsize=10)  # Cache the last 10 videos
    def _load_video_frames(self, video_file: str) -> torch.Tensor:
        """Load frames from a video file into a tensor"""
        cap = cv2.VideoCapture(os.path.join(self.root_dir, video_file))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
            
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
                
                # Resize if specified
                if self.height is not None and self.width is not None:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Convert to tensor and normalize to [0, 1]
                frame = torch.from_numpy(frame).float() / 255.0
                
                # Reorder dimensions to (C, H, W)
                frame = frame.permute(2, 0, 1)
                
                frames.append(frame)
                
                # Stop if we've reached max_frames
                if self.max_frames is not None and len(frames) >= self.max_frames:
                    break
                
            frame_idx += 1
            
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames loaded from video: {video_file}")
            
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
        video_file = self.video_metadata[idx]['file']
        
        # Load frames using the cached method
        try:
            video_tensor = self._load_video_frames(video_file)
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
        target_size=None,  # Keep original size
        max_frames=None,   # Load all frames
        frame_stride=1,    # Load every frame
        max_files=1000     # Only load 1000 videos
    )
    
    # Get a video
    video = dataset[0]
    print(f"Video shape: {video.shape}")  # Will be (n_frames, 3, height, width) 