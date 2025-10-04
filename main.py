from src.models.model_loader import preload_models_from_standard_weights
from src.pipelines import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

def setup_device():
    """Setup and return the appropriate device."""
    ALLOW_CUDA = False
    ALLOW_MPS = False
    
    if torch.cuda.is_available() and ALLOW_CUDA:
        device = "cuda"
    elif torch.backends.mps.is_available() and ALLOW_MPS:
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    return device

def load_models(checkpoint_path, device):
    """Load pretrained models."""
    print("Loading models...")
    models = preload_models_from_standard_weights(checkpoint_path, device)
    print("Models loaded successfully!")
    return models

def run_generation(
    prompt,
    uncond_prompt="",
    input_image_path=None,
    strength=0.9,
    do_cfg=True,
    cfg_scale=7.5,
    sampler="ddpm",
    num_inference_steps=50,
    seed=42,
    output_path="./outputs/generated.png"
):
    """Run the image generation pipeline."""
    
    # Setup
    device = setup_device()
    
    # Load tokenizer
    tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
    
    # Load models
    checkpoint_path = "./data/v1-5-pruned-emaonly.ckpt"
    models = load_models(checkpoint_path, device)
    
    # Load input image if provided
    input_image = None
    if input_image_path:
        input_image = Image.open(input_image_path)
        print(f"Loaded input image from: {input_image_path}")
    
    # Generate
    print("Starting generation...")
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        models=models,
        seed=seed,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    
    # Save output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(output_image).save(output_path)
    print(f"Image saved to: {output_path}")
    
    return output_image