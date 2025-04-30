from lip_sync_model.model import BiSeNet
import numpy as np
import torch
import torch.distributed as dist
from video_dataset import VideoDataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from typing import List, Tuple
import os
import signal
import atexit
import argparse

# Process the output to get the parsing map using torch operations
def get_face_mask(out):
    parsing = torch.argmax(out, dim=2)

    # Create a binary mask for lips (class 12 is upper lip, 13 is lower lip)
    lip_mask = torch.zeros_like(parsing, dtype=torch.float)
    # Facial feature classes in BiSeNet:
    # 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 6=eye_g, 7=l_ear, 8=r_ear, 9=ear_r,
    # 10=nose, 11=mouth, 12=u_lip, 13=l_lip, 14=neck, 15=neck_l, 16=cloth, 17=hair, 18=hat
    
    # Skin
    lip_mask[parsing == 1] = 1
    # Eyebrows
    lip_mask[(parsing == 2) | (parsing == 3)] = 1
    # Eyes
    lip_mask[(parsing == 4) | (parsing == 5)] = 1
    # Eye glasses
    lip_mask[parsing == 6] = 1
    # Ears
    lip_mask[(parsing == 7) | (parsing == 8)] = 1
    # Ear rings
    lip_mask[parsing == 9] = 1
    # Nose
    lip_mask[parsing == 10] = 1
    # Mouth
    lip_mask[parsing == 11] = 1
    # Lips
    lip_mask[(parsing == 12) | (parsing == 13)] = 1
    
    lip_mask_tensor = lip_mask.to(torch.uint8)

    return lip_mask_tensor


def cleanup():
    """Properly clean up distributed resources"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Process group destroyed successfully")

def signal_handler(sig, frame):
    """Handle termination signals by cleaning up first"""
    print("Received signal, cleaning up...")
    cleanup()
    exit(0)

def train(args):
    # Register cleanup functions
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Create dataset
    face_ds = VideoDataset(
        root_dir=args.root_dir,
        max_files=args.max_files,
        max_frames=args.max_frames,
        output_dir=args.output_dir
    )
    
    # Create dataloader - always batch size 1 to preserve original dimensions
    dataloader = DataLoader(
        face_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_videos
    )
    
    # Load model
    model = BiSeNet(n_classes=19)
    model.load_state_dict(torch.load("dataset/79999_iter.pth"))
    model = accelerator.prepare(model)
    print("device: ", accelerator.device)
    model.eval()
    
    # Prepare dataloader
    dataloader = accelerator.prepare(dataloader)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Process videos
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                videos, filenames = batch_data
                
                # videos shape: (batch_size, n_frames, 3, H, W)
                out = model(videos.to(accelerator.device))[0]

                mask_tensor = get_face_mask(out)
                
                # Save lip mask for this video
                mask_filename = os.path.splitext(filenames[0])[0] + "_mask.pt"
                torch.save(mask_tensor, os.path.join(args.output_dir, mask_filename))
                # print(f"Saved mask for {mask_filename} with shape {mask_tensor.shape}")
                
    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        # Ensure cleanup happens even if there's an exception
        cleanup()

def collate_videos(batch):
    """
    Custom collate function for DataLoader to handle (tensor, filename) tuples
    """
    videos, filenames = zip(*batch)
    videos = torch.stack(videos)
    return videos, filenames

def parse_args():
    parser = argparse.ArgumentParser(description='Create lip masks from videos')
    parser.add_argument('--root_dir', type=str, default='dataset/train', 
                        help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='lip_masks', 
                        help='Directory to save lip masks')
    parser.add_argument('--max_files', type=int, default=None, 
                        help='Maximum number of videos to process')
    parser.add_argument('--max_frames', type=int, default=None, 
                        help='Maximum number of frames to process')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of dataloader workers')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Running with configuration:")
    print(f"  root_dir: {args.root_dir}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  max_files: {args.max_files}")
    print(f"  max_frames: {args.max_frames}")
    print(f"  num_workers: {args.num_workers}")
    train(args)

    # accelerate launch create_lip_masks.py \
    # --num_workers 4 \
    # --max_frames 100 \
    # --output_dir dataset/lip_masks
