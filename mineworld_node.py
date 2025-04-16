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
MINEWORLD_CHECKPOINTS_DIR = os.path.join(MINEWORLD_MODEL_DIR, "checkpoints")
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
        # Create mineworld model directory and checkpoints directory if they don't exist
        os.makedirs(MINEWORLD_MODEL_DIR, exist_ok=True)
        os.makedirs(MINEWORLD_CHECKPOINTS_DIR, exist_ok=True)
        
        # Check for model checkpoint file
        model_filename = f"{model_name}.ckpt"
        local_model_path = os.path.join(MINEWORLD_CHECKPOINTS_DIR, model_filename)
        
        # Create config directory
        config_dir = os.path.join(MINEWORLD_MODEL_DIR, "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        # Config path
        config_filename = f"{model_name}.yaml"
        local_config_path = os.path.join(config_dir, config_filename)
        
        # VAE paths
        vae_dir = os.path.join(MINEWORLD_CHECKPOINTS_DIR, "vae")
        os.makedirs(vae_dir, exist_ok=True)
        local_vae_path = os.path.join(vae_dir, "vae.ckpt")
        local_vae_config_path = os.path.join(vae_dir, "config.json")
        
        # Download model if needed
        if not os.path.exists(local_model_path):
            print(f"Model {model_name} not found. Downloading from HuggingFace...")
            try:
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
        if download_vae:
            print("Checking for VAE model and config...")
            try:
                # Always download VAE config first
                if not os.path.exists(local_vae_config_path):
                    print("Downloading VAE config...")
                    huggingface_hub.hf_hub_download(
                        repo_id=HF_REPO_ID,
                        filename="checkpoints/vae/config.json",
                        local_dir=MINEWORLD_CHECKPOINTS_DIR,
                        local_dir_use_symlinks=False
                    )
                    print(f"Successfully downloaded VAE config to {local_vae_config_path}")
                
                # Then download VAE checkpoint
                if not os.path.exists(local_vae_path):
                    print("Downloading VAE checkpoint...")
                    huggingface_hub.hf_hub_download(
                        repo_id=HF_REPO_ID,
                        filename="checkpoints/vae/vae.ckpt",
                        local_dir=MINEWORLD_CHECKPOINTS_DIR,
                        local_dir_use_symlinks=False
                    )
                    print(f"Successfully downloaded VAE checkpoint to {local_vae_path}")
            except Exception as e:
                print(f"Error downloading VAE: {e}")
                if not os.path.exists(local_vae_config_path):
                    print("WARNING: VAE config is missing! This will cause errors.")
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
        vae_config_path = os.path.join(MINEWORLD_CHECKPOINTS_DIR, "vae", "config.json")
        vae_ckpt_path = os.path.join(MINEWORLD_CHECKPOINTS_DIR, "vae", "vae.ckpt")
        
        # Verify VAE config file exists
        if not os.path.exists(vae_config_path):
            print(f"ERROR: VAE config file not found at {vae_config_path}")
            print("Attempting to redownload VAE config...")
            try:
                huggingface_hub.hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename="checkpoints/vae/config.json",
                    local_dir=MINEWORLD_CHECKPOINTS_DIR,
                    local_dir_use_symlinks=False
                )
                print(f"Successfully downloaded VAE config to {vae_config_path}")
            except Exception as e:
                print(f"Error downloading VAE config: {e}")
                raise ValueError(f"Could not download VAE config file to {vae_config_path}")
                
        # Verify VAE checkpoint exists
        if not os.path.exists(vae_ckpt_path):
            print(f"ERROR: VAE checkpoint not found at {vae_ckpt_path}")
            print("Attempting to redownload VAE checkpoint...")
            try:
                huggingface_hub.hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename="checkpoints/vae/vae.ckpt",
                    local_dir=MINEWORLD_CHECKPOINTS_DIR,
                    local_dir_use_symlinks=False
                )
                print(f"Successfully downloaded VAE checkpoint to {vae_ckpt_path}")
            except Exception as e:
                print(f"Error downloading VAE checkpoint: {e}")
                raise ValueError(f"Could not download VAE checkpoint file to {vae_ckpt_path}")
        
        # Update the config directly with string values, avoiding None values
        # From analyzing vae.py, we know it expects direct strings for config_path and ckpt_path
        if hasattr(config, 'model') and hasattr(config.model, 'params') and \
           hasattr(config.model.params, 'tokenizer_config'):
            
            # Ensure params exists
            if not hasattr(config.model.params.tokenizer_config, 'params'):
                config.model.params.tokenizer_config.params = {}
            
            # Set the paths to string values
            config.model.params.tokenizer_config.params.config_path = str(vae_config_path)
            config.model.params.tokenizer_config.params.ckpt_path = str(vae_ckpt_path)
            
            print(f"Updated VAE config path in OmegaConf to: {vae_config_path}")
            print(f"Updated VAE ckpt path in OmegaConf to: {vae_ckpt_path}")
        else:
            print("WARNING: Could not find expected path to update VAE config in OmegaConf.")
            raise ValueError("Could not update VAE config and checkpoint paths in model configuration")
        
        # Print config for debugging
        print("VAE config params:")
        print(f"config_path = {config.model.params.tokenizer_config.params.config_path}")
        print(f"ckpt_path = {config.model.params.tokenizer_config.params.ckpt_path}")
        
        # Load model
        try:
            model = load_model(config, model_path, gpu=True, eval_mode=True)
            print(f"Loaded MineWorld model: {model_size}")
            return (model,)
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise

class MineWorldInitialState:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MINEWORLD_MODEL",),
                # Infer num_init_frames from image batch size.
                "image": ("IMAGE", { "tooltip": "Input image batch (BHWC format). Batch size determines the number of initial frames used for priming (e.g., 4-8 recommended)." }),
                # Number of actions must be batch_size - 1.
                "initial_actions": ("STRING", {
                    "multiline": True, 
                    "default": "forward\nforward\nforward", 
                    "tooltip": "Actions corresponding to transitions between initial frames. Provide BATCH_SIZE - 1 lines. Format: 'action_name' or 'action_name:cam_y,cam_x'."
                })
            },
        }
    
    RETURN_TYPES = ("MINEWORLD_STATE",)
    RETURN_NAMES = ("state",)
    FUNCTION = "initialize_state"
    CATEGORY = "MineWorld"

    def parse_action_string(self, action_str, mc_dataset):
        """Parses a single action string into an action dict."""
        action_dict = NOOP_ACTION.copy()
        action_dict['camera'] = np.array([0, 0]) # Ensure camera is initialized
        
        parts = action_str.strip().split(':')
        action_name = parts[0]
        
        # Check against NOOP_ACTION keys first for basic actions
        if action_name in NOOP_ACTION:
             action_dict[action_name] = 1
        # Then check against the action vocab (might include specific token names)
        elif hasattr(mc_dataset, 'action_vocab') and action_name in mc_dataset.action_vocab:
            action_dict[action_name] = 1
        else:
            # If not found anywhere, it's likely invalid
            raise ValueError(f"Invalid action name: '{action_name}'")

        if len(parts) > 1:
            try:
                cam_parts = parts[1].split(',')
                cam_y = int(cam_parts[0])
                cam_x = int(cam_parts[1])
                action_dict['camera'] = np.array([cam_y, cam_x])
            except Exception as e:
                raise ValueError(f"Invalid camera format in action '{action_str}'. Expected 'action:y,x'. Error: {e}")
                
        return action_dict

    def initialize_state(self, model, image, initial_actions):
        # --- 1. Input Validation and Preparation ---
        num_init_frames = image.shape[0] # Get number of frames from batch size
        if num_init_frames < 1:
             raise ValueError("Input image batch must contain at least 1 frame.")
        print(f"Initializing with {num_init_frames} frames from input batch.")
        
        image_batch = image # Use the whole batch (BHWC format)
        
        parsed_actions = []
        mc_dataset = MCDataset() # For parsing and tokenizing actions
        
        if num_init_frames > 1:
            action_lines = [line for line in initial_actions.strip().split('\n') if line.strip()] # Split by newline
            if len(action_lines) != num_init_frames - 1:
                raise ValueError(f"Expected {num_init_frames - 1} lines in initial_actions (image batch size - 1), but got {len(action_lines)}")
            try:
                # Ensure action vocab is built before parsing
                mc_dataset.make_action_vocab(action_vocab_offset=8192) 
                parsed_actions = [self.parse_action_string(line, mc_dataset) for line in action_lines]
            except ValueError as e:
                raise ValueError(f"Error parsing initial_actions: {e}")
        
        # --- 2. Image Preprocessing ---
        frames_list_bchw = []
        for i in range(num_init_frames):
            img_pil = Image.fromarray((image_batch[i] * 255).to(torch.uint8).cpu().numpy())
            img_pil = img_pil.resize(AGENT_RESOLUTION)
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            img_tensor_chw = torch.from_numpy(img_np).permute(2, 0, 1)
            img_tensor_chw = (img_tensor_chw * 2.0) - 1.0 # Normalize to [-1, 1]
            frames_list_bchw.append(img_tensor_chw)
            
        frames_tensor_bchw = torch.stack(frames_list_bchw).cuda() # Batch C H W
        print(f"Prepared initial frames tensor shape: {frames_tensor_bchw.shape}")

        # --- 3. Tokenization --- 
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            try:
                # Tokenize images
                images_token = model.tokenizer.tokenize_images(frames_tensor_bchw)
                
                # Reshape if necessary - Adapt rearrange based on observed output shape
                # Original expected: (B, TOKEN_PER_IMAGE)
                # Observed in mineworld.py: (B*T, H*W) -> (B, T*H*W) where T=1?
                expected_shape = (num_init_frames, TOKEN_PER_IMAGE)
                if images_token.shape != expected_shape:
                    print(f"Warning: Initial token shape {images_token.shape} differs from expected {expected_shape}. Attempting rearrange.")
                    try:
                        # Assuming the tokenizer might output (B*1, H, W) which gets flattened differently
                        # Let's try the rearrange logic assuming t=1 frame per tokenization call for the batch
                        images_token = rearrange(images_token, '(b t) h w -> b (t h w)', b=num_init_frames, t=1) 
                        print(f"Tokenized images reshaped to: {images_token.shape}")
                        if images_token.shape != expected_shape:
                             raise ValueError(f"Token shape after rearrange {images_token.shape} still not expected {expected_shape}")
                    except Exception as e_rearrange:
                        raise ValueError(f"Failed to tokenize/reshape images to expected shape {expected_shape}. Got {images_token.shape}. Error: {e_rearrange}")
                else:
                     print(f"Tokenized images shape: {images_token.shape}")

                frame_tokens_list = list(torch.split(images_token, 1, dim=0)) # List of (1, TOKEN_PER_IMAGE)
                
                # Tokenize actions
                action_tokens_list = []
                if num_init_frames > 1:
                    for action_dict in parsed_actions:
                        action_indices = mc_dataset.get_action_index_from_actiondict(action_dict, action_vocab_offset=8192)
                        action_tensor = torch.tensor([action_indices]).cuda()
                        action_tokens_list.append(action_tensor)
                        # print(f"Tokenized action tensor shape: {action_tensor.shape}") # Less verbose

                # --- 4. Interleave and Prime KV Cache --- 
                interleaved_tokens = []
                for i in range(num_init_frames - 1):
                    interleaved_tokens.append(frame_tokens_list[i])
                    interleaved_tokens.append(action_tokens_list[i])
                interleaved_tokens.append(frame_tokens_list[-1])
                
                _vis_act = torch.cat(interleaved_tokens, dim=1) # Concatenate along sequence dim
                print(f"Priming sequence shape: {_vis_act.shape}")
                
                print("Priming KV cache with initial sequence...")
                model.transformer.refresh_kvcache() # Ensure cache is clear
                
                # Use prefill_for_gradio for priming as in mineworld.py
                _, last_pos = model.transformer.prefill_for_gradio(_vis_act)
                
                print(f"KV cache primed. Last position ID: {last_pos}")

                # --- 5. Create Initial State --- 
                state = {
                    "image_tokens": frame_tokens_list[-1], # Tokens of the *last* initial frame
                    "last_pos": last_pos, 
                    "frame_history": frames_list_bchw, # History of actual image tensors (BCHW)
                    "token_history": frame_tokens_list, # History of image tokens (List of [1, TOKEN_PER_IMAGE])
                }
                
                return (state,)
            except Exception as e:
                print(f"ERROR during initial state creation or KV cache priming: {e}")
                import traceback
                traceback.print_exc()
                raise ValueError(f"Failed to initialize state or prime KV cache: {e}")

# Helper: Define NOOP_ACTION here if not imported (needed for parse_action_string)
NOOP_ACTION = {
    "forward": 0, "back": 0, "left": 0, "right": 0,
    "jump": 0, "attack": 0, "use": 0, "pickItem": 0,
    "drop": 0, "sneak": 0, "sprint": 0, "swapHands": 0,
    "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0, 
    "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0,
    "camera": np.array([0, 0]), "ESC": 0, "inventory": 0 # Add ESC/inventory if needed by parser
}

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
        action_tensor = torch.tensor([action_index]).cuda()
        
        # Make sure we're using the correct token shape from previous state
        input_tokens = state["image_tokens"]
        
        # Ensure input tokens are in correct shape (B, L) where B is batch size, L is length
        if len(input_tokens.shape) != 2:
            input_tokens = input_tokens.view(1, -1)  # Reshape to (1, L)
        
        # Debug print tensor shapes
        print(f"Input tokens shape: {input_tokens.shape}")
        print(f"Action tensor shape: {action_tensor.shape}")
        print(f"Last position: {state['last_pos']}")
        
        # Generate next frame
        try:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                # Allow torch.compile from lvm.py to work
                next_tokens, last_pos = model.transformer.decode_img_token_for_gradio(
                    input_action=action_tensor, 
                    position_id=state["last_pos"],
                    max_new_tokens=TOKEN_PER_IMAGE + 1
                )
            
            next_tokens = torch.cat(next_tokens, dim=-1).cuda()
            
            # Decode token to image
            next_image = model.tokenizer.token2image(next_tokens)
            # Create tensor in BCHW format, normalize later if needed for history
            next_image_tensor = torch.from_numpy(next_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0 
            next_image_tensor_norm = (next_image_tensor * 2.0) - 1.0 # Normalize for history if needed
            
            # Update state
            new_state = {
                "image_tokens": next_tokens,
                "last_pos": last_pos.item(), # Convert returned tensor to int
                "frame_history": state["frame_history"] + [next_image_tensor_norm],
                "token_history": state["token_history"] + [next_tokens],
            }
            
            # Convert to ComfyUI image format (BHWC)
            output_image = next_image_tensor.permute(0, 2, 3, 1) # Use non-normalized tensor for output
            
            return (new_state, output_image)
        except Exception as e:
            print(f"ERROR during action application: {e}")
            import traceback
            traceback.print_exc()
            
            # Fall back to returning the original image from state
            if len(state["frame_history"]) > 0:
                last_frame = state["frame_history"][-1]
                # Convert to ComfyUI image format (BHWC) if needed
                if last_frame.shape[1] == 3:  # If in BCHW format
                    output_image = last_frame.permute(0, 2, 3, 1)
                else:
                    output_image = last_frame
                return (state, output_image)  # Return same state with last image
            else:
                raise ValueError("Failed to generate frame and no previous frames available.")

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
            action_tensor = torch.tensor([action_index]).cuda()
            
            # Make sure we're using the correct token shape from previous state
            input_tokens = current_state["image_tokens"]
            
            # Ensure input tokens are in correct shape (B, L) where B is batch size, L is length
            if len(input_tokens.shape) != 2:
                input_tokens = input_tokens.view(1, -1)  # Reshape to (1, L)
            
            # Debug print the tensor shapes
            print(f"Input tokens shape: {input_tokens.shape}")
            print(f"Action tensor shape: {action_tensor.shape}")
            print(f"Last position: {current_state['last_pos']}")
            
            # Generate next frame
            try:
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    # Allow torch.compile from lvm.py to work
                    next_tokens, last_pos = model.transformer.decode_img_token_for_gradio(
                        input_action=action_tensor, 
                        position_id=current_state["last_pos"],
                        max_new_tokens=TOKEN_PER_IMAGE + 1
                    )
                
                next_tokens = torch.cat(next_tokens, dim=-1).cuda()
                
                # Decode token to image
                next_image = model.tokenizer.token2image(next_tokens)
                # Create tensor in HWC format for output list
                next_image_tensor_hwc = torch.from_numpy(next_image).float() / 255.0 
                 # Create tensor in BCHW format normalized for history
                next_image_tensor_bchw_norm = next_image_tensor_hwc.permute(2, 0, 1).unsqueeze(0)
                next_image_tensor_bchw_norm = (next_image_tensor_bchw_norm * 2.0) - 1.0
                
                # Add to output frames (HWC, range 0-1)
                output_frames.append(next_image_tensor_hwc)
                
                # Update state
                current_state = {
                    "image_tokens": next_tokens,
                    "last_pos": last_pos.item(), # Convert returned tensor to int
                    "frame_history": current_state["frame_history"] + [next_image_tensor_bchw_norm],
                    "token_history": current_state["token_history"] + [next_tokens],
                }
            except ValueError as e:
                print(f"ERROR during generation: {e}")
                import traceback
                traceback.print_exc()
                
                if len(output_frames) == 0:
                    # If we haven't generated any frames yet, we're in trouble
                    raise ValueError("Failed to generate first frame. Cannot continue.")
                else:
                    # If we have at least one frame, duplicate the last frame we successfully generated
                    print(f"Duplicating last successful frame instead of failing...")
                    last_frame = output_frames[-1]
                    output_frames.append(last_frame)
                    # Don't update the state - we'll try to continue from the last successful state
        
        # Stack frames and convert to BHWC format for ComfyUI
        output_tensor = torch.stack(output_frames, dim=0) # Stack HWC tensors -> BHWC
        
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