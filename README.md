# WORK IN PROGRESS


# ComfyUI MineWorld Nodes

This extension integrates Microsoft's [MineWorld](https://github.com/microsoft/mineworld) - an interactive world model for Minecraft - into ComfyUI.

MineWorld allows you to generate interactive Minecraft gameplay based on actions you provide, creating realistic Minecraft gameplay videos.

## Requirements

- Python 3.10
- CUDA-compatible GPU (A100/H100 recommended)
- ComfyUI
- huggingface_hub library (`pip install huggingface_hub`)

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/YOUR_USERNAME/ComfyUI-Mineworld.git
   ```

2. Install the required dependencies:
   ```bash
   cd ComfyUI-Mineworld/MineWorld
   pip install -r requirements.txt
   pip install huggingface_hub
   ```

3. Restart ComfyUI and the extension will automatically download models as needed.

## Models

The extension will automatically download models from the [official HuggingFace repo](https://huggingface.co/microsoft/mineworld) when they are first used. The following models are available:

- 300M_16f (smallest but fastest)
- 700M_16f (recommended)
- 700M_32f (longer context)
- 1200M_16f (higher quality)
- 1200M_32f (highest quality, longest context)

Models will be downloaded to your ComfyUI models directory under a "mineworld" folder.

## Usage

This extension provides several nodes for working with MineWorld:

### Load MineWorld Model

Loads the MineWorld model and prepares it for inference. The first time you use this node, it will automatically download the required model and VAE from HuggingFace.

**Inputs:**
- `model_size`: Size of the model to load (300M_16f, 700M_16f, 700M_32f, 1200M_16f, or 1200M_32f)
- `download_vae`: Whether to download the VAE model if not present (default: true)

**Outputs:**
- `model`: The loaded MineWorld model

### Initialize MineWorld State

Creates an initial state from an input image to start the Minecraft gameplay.

**Inputs:**
- `model`: The MineWorld model
- `image`: An input image (initial Minecraft scene)

**Outputs:**
- `state`: The MineWorld state

### Apply MineWorld Action

Takes a current state and applies a single action to generate the next frame.

**Inputs:**
- `model`: The MineWorld model
- `state`: Current MineWorld state
- `action`: The action to apply (forward, back, left, right, etc.)
- `camera_x`: Camera rotation on X axis (-90 to 90)
- `camera_y`: Camera rotation on Y axis (-90 to 90)

**Outputs:**
- `new_state`: Updated state after applying the action
- `image`: Generated image showing the result of the action

### Generate MineWorld Sequence

Generates multiple frames from a sequence of actions.

**Inputs:**
- `model`: The MineWorld model
- `state`: Starting MineWorld state
- `num_frames`: Number of frames to generate
- `actions`: Comma-separated list of actions to perform (example: "forward,forward,right,jump")

**Outputs:**
- `images`: Batch of generated images (can be converted to video)

## Action Format

For the Generate MineWorld Sequence node, you can specify actions with or without camera movement:

- Simple action: `forward`
- Action with camera: `forward:10,5` (where 10 is camera_x and 5 is camera_y)

## Example Workflow

1. Add a "Load MineWorld Model" node and select the model size (starts with 700M_16f)
2. Load or generate an initial Minecraft scene image
3. Connect the image and model to an "Initialize MineWorld State" node
4. Use "Apply MineWorld Action" for single steps or "Generate MineWorld Sequence" for multiple steps
5. Connect the generated images to a video output node or image viewer

## Limitations

- Requires high-end GPU for real-time performance
- Limited to Minecraft gameplay generation
- May require several seconds per frame depending on hardware

## Credits

This implementation uses [Microsoft's MineWorld](https://github.com/microsoft/mineworld), a model that was trained on Minecraft gameplay videos.

## License

This extension follows the same license as the original MineWorld repository. 