import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import io

# Load the model and use GPU if available, else use CPU
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionPipeline.from_pretrained("model", torch_dtype=torch.float32)
    return pipeline.to(device)

pipeline = load_model()

# Prompt input
prompt = st.text_input('Enter your prompt:', 'Sunset on a beach')

# Select the number of images to generate (max 5)
num_images = st.slider('How many images do you want to generate?', min_value=1, max_value=5, value=1)

# Button to generate images
generate_btn = st.button('Generate Images')

# Size options for download
size_options = {
    'Original': (None, None),  # No resizing
    'Small (256x256)': (256, 256),
    'Medium (512x512)': (512, 512),
    'Large (1024x1024)': (1024, 1024)
}

if generate_btn:
    # Check if CUDA is being used
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    st.write(f"Running on: {device_type}")

    # Generate images
    with st.spinner("Generating images..."):
        images = pipeline([prompt] * num_images).images

    # Display and provide download buttons for each image
    for i, img in enumerate(images):
        st.image(img, caption=f"Generated Image {i+1}", use_column_width=True)
        
        # Select size for download
        size_selection = st.selectbox(f"Select download size for Image {i+1}",
                                      options=list(size_options.keys()), index=0)
        width, height = size_options[size_selection]

        # Resize the image if a size other than "Original" is selected
        if width and height:
            img_resized = img.resize((width, height))
        else:
            img_resized = img  # No resizing for original size

        # Convert the image to a byte stream for download
        img_byte_arr = io.BytesIO()
        img_resized.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Create download button for each image
        st.download_button(
            label=f"Download Image {i+1} ({size_selection})",
            data=img_byte_arr,
            file_name=f"generated_image_{i+1}_{size_selection.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
