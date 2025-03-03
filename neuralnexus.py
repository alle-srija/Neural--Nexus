import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

@st.cache_resource
def load_model():
    model_id = "stabilityai/stable-diffusion-2-1"  # Use a faster model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)  
    pipe.to("cuda")  # Move to GPU
    pipe.enable_attention_slicing()  # Optimize memory
    return pipe

def generate_image(prompt, pipe):
    negative_prompt = "jewelry, blurry, bad draping, unrealistic, extra borders, wrong colors, distorted"
    image = pipe(prompt, negative_prompt=negative_prompt, height=512, width=512).images[0]
    return image

def main():
    st.title("CoutureAI: AI-Generated Fashion")
    st.write("Describe your clothing idea, and we'll generate an image!")

    pipe = load_model()
    user_input = st.text_area("Enter clothing description:", "A stylish blue saree with satin cloth and a black blouse")
    
    if st.button("Generate Image"):
        with st.spinner("Generating your design... Please wait ⏳"):
            image = generate_image(user_input, pipe)
            st.image(image, caption="Generated Clothing Design", use_column_width=True)

            # Save & Provide Download Option
            img_path = "generated_fashion.png"
            image.save(img_path)
            with open(img_path, "rb") as file:
                st.download_button(label="Download Image", data=file, file_name="fashion_design.png", mime="image/png")

if __name__ == "__main__":
    main()