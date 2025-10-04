import streamlit as st
from PIL import Image
from pathlib import Path
import time

# Import from your main.py
from main import run_generation


def main():
    st.set_page_config(page_title="Stable Diffusion Generator", layout="wide")
    
    st.title("Stable Diffusion Image Generator")
    
    # Sidebar for parameters
    st.sidebar.header("Generation Parameters")
    
    # Mode selection
    mode = st.sidebar.radio("Generation Mode", ["Text-to-Image", "Image-to-Image"])
    
    # Common parameters
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=999999, value=42)
    num_steps = st.sidebar.slider("Inference Steps", min_value=10, max_value=100, value=50, step=5)
    cfg_scale = st.sidebar.slider("CFG Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
    
    # Image-to-image specific
    strength = 0.9
    if mode == "Image-to-Image":
        strength = st.sidebar.slider("Denoising Strength", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input")
        
        # Prompt
        prompt = st.text_area(
            "Prompt",
            value="A beautiful landscape with mountains and a lake",
            height=100
        )
        
        # Negative prompt
        uncond_prompt = st.text_area(
            "Negative Prompt (Optional)",
            value="",
            height=80
        )
        
        # Image upload for img2img
        input_image = None
        uploaded_file = None
        saved_image_path = None
        
        if mode == "Image-to-Image":
            uploaded_file = st.file_uploader("Upload Input Image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file)
                st.image(input_image, caption="Input Image", use_container_width=True)
                
                # Save to images folder
                images_dir = Path("./images")
                images_dir.mkdir(exist_ok=True)
                saved_image_path = images_dir / uploaded_file.name
                input_image.save(saved_image_path)
    
    with col2:
        st.header("Output")
        output_placeholder = st.empty()
    
    # Generate button
    if st.button("Generate Image", type="primary", use_container_width=True):
        
        if mode == "Image-to-Image" and uploaded_file is None:
            st.error("Please upload an input image for Image-to-Image mode!")
            return
        
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading models and preparing generation...")
            progress_bar.progress(20)
            
            # Prepare output path
            output_dir = Path("./outputs")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"generated_{int(time.time())}.png"
            
            status_text.text(f"Generating image ({num_steps} steps)...")
            progress_bar.progress(40)
            
            # Call your run_generation function
            start_time = time.time()
            
            output_image = run_generation(
                prompt=prompt,
                uncond_prompt=uncond_prompt,
                input_image_path=str(saved_image_path) if saved_image_path else None,
                strength=strength,
                do_cfg=True,
                cfg_scale=cfg_scale,
                sampler="ddpm",
                num_inference_steps=num_steps,
                seed=seed,
                output_path=str(output_path)
            )
            
            progress_bar.progress(100)
            elapsed_time = time.time() - start_time
            
            status_text.text(f"Generation complete! ({elapsed_time:.2f}s)")
            
            # Display result
            output_placeholder.image(output_image, caption="Generated Image", use_container_width=True)
            
            # Download button
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name=output_path.name,
                    mime="image/png",
                    use_container_width=True
                )
            
            st.success(f"Image saved to: {output_path}")
            
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()