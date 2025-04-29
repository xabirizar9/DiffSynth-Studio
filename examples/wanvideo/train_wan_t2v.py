import torch, os, imageio, argparse
from pathlib import Path
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np



class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        train_dir = Path(base_path) / "train"

        # 1) list all the already-generated .tensors.pth files once
        self.processed = {
            p.name  # get original filename by splitting at the dot
            for p in train_dir.glob("*.tensors.pth")
        }

        # 2) filter metadata by file_name not in processed
        all_fnames = metadata["file_name"].to_list()
        all_texts  = metadata["text"].to_list()

        self.path = []
        self.text = []

        for fname, txt in zip(all_fnames, all_texts):
            stem = fname + ".tensors.pth"
            if stem not in self.processed:
                self.path.append(str(train_dir / fname))
                self.text.append(txt)

        print(
            f"Found {len(self.processed)} already-processed videos, "
            f"{len(self.path)} left to process."
        )

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.skipped_videos = []
        self.count = 0
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            # Track skipped videos due to insufficient frames
            self.skipped_videos.append(file_path)
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
        first_frame = np.array(first_frame)

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]

        self.count += 1

        if self.count % 100 == 0:
            print("Count: ", self.count)
            print("Skipped videos: ", len(self.skipped_videos))
        
        # Handle image files
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
            video = self.load_image(path)
        else:
            # Handle video files
            video = self.load_video(path)
            # Skip videos with insufficient frames

            if video is None:
                # Find the next valid item or fall back to a previously successful one
                for i in range(1, len(self)):
                    alt_id = (data_id + i) % len(self)
                    alt_path = self.path[alt_id]
                    if self.is_image(alt_path):
                        if not self.is_i2v:
                            return self.__getitem__(alt_id)
                    else:
                        alt_video = self.load_video(alt_path)
                        if alt_video is not None:
                            print(alt_video.shape)
                            return self.__getitem__(alt_id)
                
                # If we couldn't find a valid item, raise an error
                raise RuntimeError(f"Could not find any valid videos with sufficient frames after checking {len(self)} items")
                
        if self.is_i2v:
            video, first_frame = video
            data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
        else:
            data = {"text": text, "video": video, "path": path}
        return data
    
    def save_skipped_videos(self, output_path):
        """Save the list of skipped videos to a file"""
        with open(output_path, 'w') as f:
            for video_path in self.skipped_videos:
                f.write(f"{video_path}\n")
        print(f"Saved {len(self.skipped_videos)} skipped videos to {output_path}")
    

    def __len__(self):
        return len(self.path)



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)
            else:
                image_emb = {}
            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
            torch.save(data, path + ".tensors.pth")



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch, lip_masks_path=None, frames: int = 81):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.frames = frames
        print(len(self.path), "videos in metadata.")
        
        # Filter for tensors that exist
        self.tensor_paths = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        print(len(self.tensor_paths), "tensors cached in metadata.")
        assert len(self.tensor_paths) > 0
        
        # Handle lip masks
        self.lip_masks_path = lip_masks_path
        self.use_lip_masks = lip_masks_path is not None
        if self.use_lip_masks:
            self.lip_mask_paths = []
            valid_tensor_paths = []
            
            for tensor_path in self.tensor_paths:
                base_path = tensor_path.replace(".tensors.pth", "")
                filename = os.path.basename(base_path)
                mask_filename = os.path.splitext(filename)[0] + "_mask.pt"
                mask_path = os.path.join(lip_masks_path, mask_filename)
                
                if os.path.exists(mask_path):
                    self.lip_mask_paths.append(mask_path)
                    valid_tensor_paths.append(tensor_path)
            
            # Update tensor_paths to only include those with corresponding lip masks
            self.tensor_paths = valid_tensor_paths
            print(f"{len(self.lip_mask_paths)} lip masks found and matched with tensors.")
            print(f"{len(self.tensor_paths)} tensors remaining after filtering for lip masks.")
            assert len(self.tensor_paths) > 0, "No tensors with matching lip masks found"
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.tensor_paths), (1,))[0]
        data_id = (data_id + index) % len(self.tensor_paths) # For fixed seed.
        path = self.tensor_paths[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        data['latents'] = data['latents'][:self.frames]
        
        # Load lip mask if available
        if self.use_lip_masks:
            lip_mask = torch.load(self.lip_mask_paths[data_id], map_location="cpu")
            data["lip_mask"] = lip_mask
        
        return data
    

    def __len__(self):
        return self.steps_per_epoch



class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        vae_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        use_lip_weighted_loss=False,
        lip_alpha=5.0,
        pixel_weight=0.1
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path, vae_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path, vae_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        
        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        # Lip sync parameters
        self.use_lip_weighted_loss = use_lip_weighted_loss
        self.lip_alpha = lip_alpha
        self.pixel_weight = pixel_weight
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        
        # Explicitly ensure VAE is frozen
        if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
            self.pipe.vae.requires_grad_(False)
            self.pipe.vae.eval()
        
        # Only set denoising model to train mode
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)
        
        # Get lip mask if available
        lip_mask = batch.get("lip_mask", [None])[0]
        use_lip_loss = self.use_lip_weighted_loss and lip_mask is not None
        
        if use_lip_loss:
            lip_mask = lip_mask.to(self.device)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)
        training_target = self.scheduler.training_target(latents, noise, timestep)

        # Compute noise prediction
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        
        # Standard diffusion loss
        diff_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        diff_loss = diff_loss * self.scheduler.training_weight(timestep)
        
        # Add pixel-space lip loss if mask is available
        if use_lip_loss:
            pred_latent = noise_pred
            
            # Decode to pixel space (with no grad to save memory)
            with torch.no_grad():
                # Use the correct decoding method from the pipeline
                pred_pixels = self.pipe.vae.decode(pred_latent, device=self.device)
                target_pixels = self.pipe.vae.decode(latents, device=self.device)
                
                # Get the number of frames in original latents (before VAE decoding)
                original_num_frames = latents.shape[2]
                
                # Check if VAE output has more frames than input and truncate if needed
                if pred_pixels.shape[2] > original_num_frames:
                    pred_pixels = pred_pixels[:, :, :original_num_frames]
                if target_pixels.shape[2] > original_num_frames:
                    target_pixels = target_pixels[:, :, :original_num_frames]
                
                # Convert from [-1,1] to [0,1] range if needed
                if pred_pixels.min() < 0:
                    pred_pixels = (pred_pixels + 1) / 2
                    target_pixels = (target_pixels + 1) / 2
            
            # Simplify lip mask handling - our lip_mask shape is [frames, H, W]
            # First, add batch dimension if needed
            if lip_mask.dim() == 3:  # [frames, H, W]
                # Add batch dimension for batch processing
                lip_mask = lip_mask.unsqueeze(0)  # [1, frames, H, W]
            
            # Make sure we have the right number of frames (truncate or pad)
            if lip_mask.shape[1] != pred_pixels.shape[2]:
                if lip_mask.shape[1] > pred_pixels.shape[2]:
                    # Too many frames, truncate
                    lip_mask = lip_mask[:, :pred_pixels.shape[2]]
                else:
                    # Too few frames, repeat the last frame
                    frames_to_add = pred_pixels.shape[2] - lip_mask.shape[1]
                    last_frame = lip_mask[:, -1:].repeat(1, frames_to_add, 1, 1)
                    lip_mask = torch.cat([lip_mask, last_frame], dim=1)
            
            # Resize height/width if needed 
            if lip_mask.shape[2] != pred_pixels.shape[3] or lip_mask.shape[3] != pred_pixels.shape[4]:
                lip_mask = torch.nn.functional.interpolate(
                    lip_mask.reshape(-1, 1, lip_mask.shape[2], lip_mask.shape[3]),  # [b*frames, 1, H, W]
                    size=(pred_pixels.shape[3], pred_pixels.shape[4]),
                    mode='bilinear',
                    align_corners=False
                ).reshape(lip_mask.shape[0], lip_mask.shape[1], pred_pixels.shape[3], pred_pixels.shape[4])
            
            # Add channel dimension to match pred_pixels [B, C, F, H, W]
            lip_mask = lip_mask.unsqueeze(1)  # [B, 1, F, H, W]
            
            # Create weighted mask (1 + alpha * lip_mask)
            lip_weight = 1.0 + self.lip_alpha * lip_mask
            
            # pred_pixels is [B, C, F, H, W], average over channels for loss
            pixel_error = torch.nn.functional.l1_loss(
                pred_pixels, target_pixels, reduction='none'
            ).mean(dim=1, keepdim=True)  # Now [B, 1, F, H, W]
            
            # Weighted pixel loss
            lip_loss = (lip_weight * pixel_error).sum() / lip_weight.sum()
            
            # Combine losses
            loss = diff_loss + self.pixel_weight * lip_loss
            
            # Log both losses
            self.log("diff_loss", diff_loss.item(), prog_bar=True)
            self.log("lip_loss", lip_loss.item(), prog_bar=True)
        else:
            loss = diff_loss

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="Whether to use Weights & Biases logger.",
    )
    parser.add_argument(
        "--wandb_project",
        default="diffsynth-studio",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_name",
        default="wan",
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--lip_masks_path",
        type=str,
        default=None,
        help="Path to the directory containing lip masks",
    )
    parser.add_argument(
        "--use_lip_weighted_loss",
        action="store_true",
        default=False,
        help="Whether to use lip-weighted pixel-space loss",
    )
    parser.add_argument(
        "--lip_alpha",
        type=float,
        default=5.0,
        help="Alpha multiplier for lip regions in mask (higher = more emphasis on lips)",
    )
    parser.add_argument(
        "--pixel_weight",
        type=float,
        default=0.1,
        help="Weight for the pixel-space loss relative to diffusion loss",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    # Save the list of skipped videos to a file
    skipped_videos_path = os.path.join(args.output_path, "skipped_videos.txt")
    dataset.save_skipped_videos(skipped_videos_path)
    
    # Report final processing summary
    print("\n===== Processing Summary =====")
    print(f"Videos processed in this run: {len(dataset.path)}")
    print("===============================")
    
    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
        lip_masks_path=args.lip_masks_path,
        frames=50,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        use_lip_weighted_loss=args.use_lip_weighted_loss,
        lip_alpha=args.lip_alpha,
        pixel_weight=args.pixel_weight,
    )
    loggers = []
    
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        loggers.append(swanlab_logger)
    
    if args.use_wandb:
        from lightning.pytorch.loggers import WandbLogger
        wandb_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        wandb_config.update(vars(args))
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            log_model=True,
            save_dir=os.path.join(args.output_path, "wandb"),
            config=wandb_config
        )
        loggers.append(wandb_logger)
    
    logger = loggers if loggers else None
        
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
