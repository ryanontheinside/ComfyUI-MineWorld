import os
import sys
import torch
import numpy as np
from torch import autocast
from PIL import Image
from einops import rearrange
import requests
import tqdm
import json
import shutil
import importlib.util

# Suppress TorchDynamo errors and fall back to eager execution
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Check for huggingface_hub and install if missing
try:
    import huggingface_hub
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    import huggingface_hub

try:
    import folder_paths
except ImportError:
    # Fallback to current directory if folder_paths is not available
    class FolderPaths:
        def __init__(self):
            self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    folder_paths = FolderPaths()

# Get path to MineWorld directory and add to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
mineworld_path = os.path.join(current_dir, "MineWorld")
sys.path.append(mineworld_path)  # Simple approach - add to Python path

# Now import MineWorld components normally - using relative path
from mw_utils import load_model  # Use the file directly, not from MineWorld module
from mcdataset import MCDataset   # Use the file directly, not from MineWorld module
from omegaconf import OmegaConf

# Constants
TOKEN_PER_IMAGE = 336
AGENT_RESOLUTION = (384, 224)
MODELS_DIR = folder_paths.models_dir
MINEWORLD_MODEL_DIR = os.path.join(MODELS_DIR, "mineworld")
HF_REPO_ID = "microsoft/mineworld"

class MineWorldModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_size": (["300M_16f", "700M_16f", "700M_32f", "1200M_16f", "1200M_32f"], {"default": "700M_16f"}),
                "download_vae": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("MINEWORLD_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MineWorld"

    def ensure_model_downloaded(self, model_name, download_vae=True):
        """Ensure the model is downloaded to the ComfyUI models directory"""
        # Create mineworld model directory if it doesn't exist
        os.makedirs(MINEWORLD_MODEL_DIR, exist_ok=True)
        
        # Check for model checkpoint file
        model_filename = f"{model_name}.ckpt"
        local_model_path = os.path.join(MINEWORLD_MODEL_DIR, model_filename)
        
        # Create config directory
        config_dir = os.path.join(MINEWORLD_MODEL_DIR, "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        # Config path
        config_filename = f"{model_name}.yaml"
        local_config_path = os.path.join(config_dir, config_filename)
        
        # VAE paths
        vae_dir = os.path.join(MINEWORLD_MODEL_DIR, "vae")
        os.makedirs(vae_dir, exist_ok=True)
        local_vae_path = os.path.join(vae_dir, "vae.ckpt")
        local_vae_config_path = os.path.join(vae_dir, "config.json")
        
        # Download model if needed
        if not os.path.exists(local_model_path):
            print(f"Model {model_name} not found. Downloading from HuggingFace...")
            try:
                model_url = huggingface_hub.hf_hub_url(
                    repo_id=HF_REPO_ID,
                    filename=f"checkpoints/{model_filename}"
                )
                huggingface_hub.hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=f"checkpoints/{model_filename}",
                    local_dir=MINEWORLD_MODEL_DIR,
                    local_dir_use_symlinks=False
                )
                print(f"Successfully downloaded {model_name} model")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise
        
        # Download VAE if needed
        if download_vae and (not os.path.exists(local_vae_path) or not os.path.exists(local_vae_config_path)):
            print("VAE model or config not found. Downloading from HuggingFace...")
            try:
                huggingface_hub.hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename="checkpoints/vae/vae.ckpt",
                    local_dir=MINEWORLD_MODEL_DIR,
                    local_dir_use_symlinks=False
                )
                huggingface_hub.hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename="checkpoints/vae/config.json",
                    local_dir=MINEWORLD_MODEL_DIR,
                    local_dir_use_symlinks=False
                )
                print("Successfully downloaded VAE model and config")
            except Exception as e:
                print(f"Error downloading VAE: {e}")
                raise
        
        # Copy config from MineWorld directory if needed
        if not os.path.exists(local_config_path):
            print(f"Config for {model_name} not found. Copying from MineWorld...")
            try:
                # First try to find it in the MineWorld/configs directory
                mineworld_config_path = os.path.join(mineworld_path, "configs", config_filename)
                if os.path.exists(mineworld_config_path):
                    shutil.copy(mineworld_config_path, local_config_path)
                    print(f"Copied config from {mineworld_config_path}")
                else:
                    # If not found, try to download from HuggingFace
                    print(f"Config not found in MineWorld directory. Downloading from HuggingFace...")
                    try:
                        huggingface_hub.hf_hub_download(
                            repo_id=HF_REPO_ID,
                            filename=f"configs/{config_filename}",
                            local_dir=config_dir,
                            local_dir_use_symlinks=False
                        )
                        print(f"Successfully downloaded config for {model_name}")
                    except Exception as e:
                        print(f"Error downloading config: {e}")
                        raise
            except Exception as e:
                print(f"Error copying/downloading config: {e}")
                raise
        
        return local_model_path, local_config_path

    def load_model(self, model_size, download_vae=True):
        # Ensure model is downloaded to ComfyUI models directory
        model_path, config_path = self.ensure_model_downloaded(model_size, download_vae)
        
        # Load config
        config = OmegaConf.load(config_path)
        
        # Manually update VAE config path to point to the downloaded location
        vae_config_path = os.path.join(MINEWORLD_MODEL_DIR, "vae", "config.json")
        if hasattr(config, 'model') and hasattr(config.model, 'params') and \
           hasattr(config.model.params, 'tokenizer_config') and \
           hasattr(config.model.params.tokenizer_config, 'params'):
            config.model.params.tokenizer_config.params.config_path = vae_config_path
            print(f"Updated VAE config path in OmegaConf to: {vae_config_path}")
            
            # Also update the VAE checkpoint path
            vae_ckpt_path = os.path.join(MINEWORLD_MODEL_DIR, "vae", "vae.ckpt")
            if os.path.exists(vae_ckpt_path): # Check if VAE was downloaded
                config.model.params.tokenizer_config.params.ckpt_path = vae_ckpt_path
                print(f"Updated VAE ckpt path in OmegaConf to: {vae_ckpt_path}")
            else:
                # If VAE wasn't downloaded, ckpt_path might be intentionally None
                print("VAE checkpoint not found locally, leaving ckpt_path as is in config.")
        else:
            print("Warning: Could not find expected path to update VAE config in OmegaConf.")
        
        # Load model
        model = load_model(config, model_path, gpu=True, eval_mode=True)
        print(f"Loaded MineWorld model: {model_size}")

        # Return model
        return (model,)

class MineWorldInitialState:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MINEWORLD_MODEL",),
                "image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("MINEWORLD_STATE",)
    RETURN_NAMES = ("state",)
    FUNCTION = "initialize_state"
    CATEGORY = "MineWorld"

    def initialize_state(self, model, image):
        # Convert tensor to the format expected by MineWorld
        # ComfyUI image tensors are BHWC format
        # Convert to BCHW format expected by the model
        batch_size = image.shape[0]
        if batch_size > 1:
            # Take only the first image if batch size > 1
            image = image[0:1]
        
        # Resize image if needed
        image_pil = Image.fromarray((image[0] * 255).to(torch.uint8).cpu().numpy())
        image_pil = image_pil.resize(AGENT_RESOLUTION)
        
        # Convert back to tensor in BCHW format normalized to [-1, 1]
        image_np = np.array(image_pil).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        image_tensor = (image_tensor * 2.0) - 1.0 # Normalize to [-1, 1]
        
        # Tokenize the image
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            image_tokens = model.tokenizer.tokenize_images(image_tensor.cuda())
        
        # Reshape tokens to the format expected by the transformer
        image_tokens = rearrange(image_tokens, '(b t) h w -> b (t h w)', b=1)
        
        # Create initial state with tokenized image and position info
        state = {
            "image_tokens": image_tokens,
            "last_pos": 0,
            "frame_history": [image_tensor],
            "token_history": [image_tokens],
        }
        
        return (state,)

class MineWorldAction:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MINEWORLD_MODEL",),
                "state": ("MINEWORLD_STATE",),
                "action": (["forward", "back", "left", "right", "jump", "attack", "use", "drop"],),
                "camera_x": ("INT", {"default": 0, "min": -90, "max": 90}),
                "camera_y": ("INT", {"default": 0, "min": -90, "max": 90}),
            },
        }
    
    RETURN_TYPES = ("MINEWORLD_STATE", "IMAGE",)
    RETURN_NAMES = ("new_state", "image",)
    FUNCTION = "apply_action"
    CATEGORY = "MineWorld"

    def apply_action(self, model, state, action, camera_x, camera_y):
        # Create action dictionary
        action_dict = {
            "forward": 0, "back": 0, "left": 0, "right": 0,
            "jump": 0, "attack": 0, "use": 0, "drop": 0, "sneak": 0, 
            "sprint": 0, "swapHands": 0, "pickItem": 0,
            "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0, 
            "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0,
            "camera": np.array([camera_y, camera_x])
        }
        
        # Set the selected action to 1
        action_dict[action] = 1
        
        # Convert action to model format
        mc_dataset = MCDataset()
        action_index = mc_dataset.get_action_index_from_actiondict(action_dict, action_vocab_offset=8192)
        action_tensor = torch.tensor([action_index]).unsqueeze(0).cuda()
        
        # Generate next frame
        model.transformer.refresh_kvcache()
        
        # Generate with last frame and action
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16), torch._dynamo.disable():
            next_tokens, last_pos = model.transformer.decode_img_token_for_gradio(
                input_action=action_tensor, 
                position_id=state["last_pos"],
                max_new_tokens=TOKEN_PER_IMAGE + 1
            )
        
        next_tokens = torch.cat(next_tokens, dim=-1).cuda()
        
        # Decode token to image
        next_image = model.tokenizer.token2image(next_tokens)
        next_image_tensor = torch.from_numpy(next_image).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Update state
        new_state = {
            "image_tokens": next_tokens,
            "last_pos": last_pos[0],
            "frame_history": state["frame_history"] + [next_image_tensor],
            "token_history": state["token_history"] + [next_tokens],
        }
        
        # Convert to ComfyUI image format (BHWC)
        output_image = next_image_tensor.permute(0, 2, 3, 1)
        
        return (new_state, output_image)

class MineWorldGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MINEWORLD_MODEL",),
                "state": ("MINEWORLD_STATE",),
                "num_frames": ("INT", {"default": 10, "min": 1, "max": 100}),
                "actions": ("STRING", {"multiline": True, "default": "forward,forward,forward,right,right,forward"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_sequence"
    CATEGORY = "MineWorld"

    def generate_sequence(self, model, state, num_frames, actions):
        # Parse actions
        action_list = actions.strip().split(',')
        if len(action_list) < num_frames:
            # Repeat the last action if not enough actions provided
            action_list.extend([action_list[-1]] * (num_frames - len(action_list)))
        elif len(action_list) > num_frames:
            # Truncate if too many actions
            action_list = action_list[:num_frames]
        
        # Generate frames
        output_frames = []
        current_state = state
        mc_dataset = MCDataset()
        
        for action_str in action_list:
            # Parse action and camera
            if ':' in action_str:
                action_parts = action_str.split(':')
                action = action_parts[0]
                camera_parts = action_parts[1].split(',')
                camera_x = int(camera_parts[0])
                camera_y = int(camera_parts[1])
            else:
                action = action_str
                camera_x = 0
                camera_y = 0
            
            # Create action dictionary
            action_dict = {
                "forward": 0, "back": 0, "left": 0, "right": 0,
                "jump": 0, "attack": 0, "use": 0, "drop": 0, "sneak": 0, 
                "sprint": 0, "swapHands": 0, "pickItem": 0,
                "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0, 
                "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0,
                "camera": np.array([camera_y, camera_x])
            }
            
            # Set the selected action to 1
            action_dict[action] = 1
            
            # Convert action to model format
            action_index = mc_dataset.get_action_index_from_actiondict(action_dict, action_vocab_offset=8192)
            action_tensor = torch.tensor([action_index]).unsqueeze(0).cuda()
            
            # Generate next frame
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16), torch._dynamo.disable():
                next_tokens, last_pos = model.transformer.decode_img_token_for_gradio(
                    input_action=action_tensor, 
                    position_id=current_state["last_pos"],
                    max_new_tokens=TOKEN_PER_IMAGE + 1
                )
            
            next_tokens = torch.cat(next_tokens, dim=-1).cuda()
            
            # Decode token to image
            next_image = model.tokenizer.token2image(next_tokens)
            next_image_tensor = torch.from_numpy(next_image).float() / 255.0
            
            # Add to output frames
            output_frames.append(next_image_tensor)
            
            # Update state
            current_state = {
                "image_tokens": next_tokens,
                "last_pos": last_pos[0],
                "frame_history": current_state["frame_history"] + [torch.from_numpy(next_image).permute(2, 0, 1).unsqueeze(0) / 255.0],
                "token_history": current_state["token_history"] + [next_tokens],
            }
        
        # Stack frames and convert to BHWC format for ComfyUI
        output_tensor = torch.stack(output_frames, dim=0)
        
        return (output_tensor,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "MineWorldModelLoader": MineWorldModelLoader,
    "MineWorldInitialState": MineWorldInitialState,
    "MineWorldAction": MineWorldAction,
    "MineWorldGenerate": MineWorldGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MineWorldModelLoader": "Load MineWorld Model",
    "MineWorldInitialState": "Initialize MineWorld State",
    "MineWorldAction": "Apply MineWorld Action",
    "MineWorldGenerate": "Generate MineWorld Sequence",
} 