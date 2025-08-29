import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
from skimage import filters
import cv2
import io

# Configure page
st.set_page_config(
    page_title="Image Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)

def convert_array_to_pil(img_array, mode='RGB'):
    """Convert numpy array to PIL Image"""
    if len(img_array.shape) == 2:  # Grayscale
        img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1 else img_array.astype(np.uint8)
        return Image.fromarray(img_array, mode='L')
    else:  # Color
        img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1 else img_array.astype(np.uint8)
        return Image.fromarray(img_array, mode='RGB')

def create_download_link(img, filename, format_type="PNG"):
    """Create a download link for image"""
    buffer = io.BytesIO()
    img.save(buffer, format=format_type)
    buffer.seek(0)
    return buffer.getvalue()

# Main app
st.title("Image Processing Dashboard")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("Processing Controls")

# File upload
uploaded_file = st.file_uploader(
    "Upload Your Image",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
)

if uploaded_file is not None:
    try:
        # Load and display original image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Main layout with two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
            # Image information
            with st.expander("Image Information"):
                st.write(f"**Dimensions:** {image.size[0]} × {image.size[1]} pixels")
                st.write(f"**Format:** {image.format}")
                st.write(f"**Mode:** {image.mode}")
                st.write(f"**File Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
                
                if len(img_array.shape) == 3:
                    st.write(f"**Channels:** {img_array.shape[2]} (Color)")
                else:
                    st.write("**Channels:** 1 (Grayscale)")
        
        # Processing options
        st.sidebar.subheader("Processing Options")
        
        process_type = st.sidebar.selectbox(
            "Select Processing Type:",
            ["Original", "Grayscale", "Edge Detection", "Brightness Adjustment"],
            help="Choose the type of image processing to apply"
        )
        
        # Brightness adjustment parameter
        if process_type == "Brightness Adjustment":
            brightness_factor = st.sidebar.slider(
                "Brightness Factor",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="1.0 = original, <1.0 = darker, >1.0 = brighter"
            )
        
        with col2:
            st.subheader("Processed Image")
            
            # Apply selected processing
            with st.spinner("Processing image..."):
                if process_type == "Original":
                    processed_pil = image
                    processed_img = img_array
                    
                elif process_type == "Grayscale":
                    if len(img_array.shape) == 3:
                        processed_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else:
                        processed_img = img_array
                    processed_pil = convert_array_to_pil(processed_img, 'L')
                    
                elif process_type == "Edge Detection":
                    if len(img_array.shape) == 3:
                        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray_img = img_array
                    processed_img = filters.sobel(gray_img)
                    processed_pil = convert_array_to_pil(processed_img, 'L')
                    
                elif process_type == "Brightness Adjustment":
                    enhancer = ImageEnhance.Brightness(image)
                    processed_pil = enhancer.enhance(brightness_factor)
                    processed_img = np.array(processed_pil)
            
            # Display processed image
            st.image(processed_pil, use_container_width=True)
            
            # Download buttons
            if process_type != "Original":
                st.markdown("### Download Image")
                
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    png_data = create_download_link(processed_pil, "processed_image.png", "PNG")
                    st.download_button(
                        label="Download as PNG",
                        data=png_data,
                        file_name=f"processed_{process_type.lower().replace(' ', '_')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_download2:
                    if processed_pil.mode == 'RGBA':
                        jpeg_img = processed_pil.convert('RGB')
                    else:
                        jpeg_img = processed_pil
                    jpeg_data = create_download_link(jpeg_img, "processed_image.jpg", "JPEG")
                    st.download_button(
                        label="Download as JPEG",
                        data=jpeg_data,
                        file_name=f"processed_{process_type.lower().replace(' ', '_')}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
        
        # Simple statistics
        st.markdown("---")
        st.subheader("Image Statistics")
        
        # Calculate statistics for grayscale version
        if len(img_array.shape) == 3:
            gray_for_stats = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_for_stats = img_array
            
        processed_gray = processed_img
        if len(processed_img.shape) == 3:
            processed_gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
        
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.write("**Original Image:**")
            st.metric("Mean Brightness", f"{np.mean(gray_for_stats):.1f}")
            st.metric("Contrast (Std)", f"{np.std(gray_for_stats):.1f}")
        
        with stat_col2:
            if process_type != "Original":
                st.write("**Processed Image:**")
                st.metric("Mean Brightness", f"{np.mean(processed_gray):.1f}")
                st.metric("Contrast (Std)", f"{np.std(processed_gray):.1f}")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try uploading a different image or check the file format.")

else:
    st.info("Please upload an image to get started!")

st.markdown("---")
st.markdown("*Built with Streamlit | Namal University Summer School 2025*")
