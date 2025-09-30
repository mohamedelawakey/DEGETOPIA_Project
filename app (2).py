import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2


# Page configuration
st.set_page_config(
    page_title="Anemia Detection",
    page_icon="🩸",
    layout="wide"
)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('Anemia_final.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing
def preprocess_image(image, img_size=(128, 128)):
    """
    Preprocess image for model prediction
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(img_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize (same as training)
    img_array = img_array / 128.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Prediction function
def predict_anemia(model, image):
    """
    Predict anemia presence
    """
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img, verbose=0)
    probability = float(prediction[0][0])
    
    return probability

# Main application interface
def main():
    # Main title
    st.title("🩸 Anemia Detection System using AI")
    st.markdown("---")
    
    # Sidebar for information
    with st.sidebar:
        st.header("📋 System Information")
        st.info("""
        **Model Used:** EfficientNetB0
        
        **How to Use:**
        1. Choose image type (Conjunctiva, Fingernails, or Palm)
        2. Upload a clear image
        3. Wait for results
        
        **Note:** This system is for educational purposes only and does not replace medical examination.
        """)
        
        st.markdown("---")
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model not loaded. Make sure Anemia_final.keras file exists in the same folder.")
        return
    
    # Select image type
    st.subheader("📸 Select Image Type")
    image_type = st.selectbox(
        "Image Type:",
        ["Conjunctiva (Eye)", "Finger Nails", "Palm"]
    )
    
    # Upload image
    st.subheader("⬆️ Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Image should be clear and of good quality"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Analysis Result")
            
            # Analysis button
            if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        # Make prediction
                        probability = predict_anemia(model, image)
                        
                        # Display results
                        st.markdown("---")
                        
                        # Probability metric
                        st.metric(
                            label="Anemia Probability",
                            value=f"{probability * 100:.2f}%"
                        )
                        
                        # Progress bar
                        st.progress(probability)
                        
                        st.markdown("---")
                        
                        # Final diagnosis
                        if probability >= confidence_threshold:
                            st.error(f"""
                            ### ⚠️ Result: Anemia Positive
                            
                            **Probability:** {probability * 100:.2f}%
                            
                            **Recommendations:**
                            - Consult a specialist doctor immediately
                            - Perform necessary blood tests
                            - Do not rely on this diagnosis alone
                            """)
                        else:
                            st.success(f"""
                            ### ✅ Result: Anemia Negative
                            
                            **Probability:** {(1 - probability) * 100:.2f}%
                            
                            **Note:** 
                            - This is a preliminary result only
                            - It's recommended to consult a doctor
                            - Maintain a healthy diet
                            """)
                        
                        # Additional information
                        with st.expander("ℹ️ Additional Information"):
                            st.write(f"""
                            - **Image Type:** {image_type}
                            - **Original Image Size:** {image.size}
                            - **Threshold Used:** {confidence_threshold}
                            - **Prediction Model:** EfficientNetB0
                            """)
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
    
    else:
        # Guidance message
        st.info("👆 Upload an image to start analysis")
        
        # Examples section
        with st.expander("📖 Acceptable Image Examples"):
            st.write("""
            **Conjunctiva (Eye):**
            - Clear image of lower eyelid
            - Good natural lighting
            - No strong flash
            
            **Finger Nails:**
            - Clear image of fingernails
            - Neutral background
            - Clean nails
            
            **Palm:**
            - Full image of palm
            - Even lighting
            - Simple background
            """)
    
    # Anemia information section
    st.markdown("---")
    with st.expander("🏥 Information About Anemia"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Common Symptoms:
            - Persistent fatigue and tiredness
            - Pale skin and face
            - Shortness of breath
            - Dizziness and headache
            - Cold hands and feet
            - Irregular heartbeat
            """)
        
        with col2:
            st.markdown("""
            ### Causes of Anemia:
            - Iron deficiency in diet
            - Vitamin B12 deficiency
            - Chronic blood loss
            - Chronic diseases
            - Genetic disorders
            - Pregnancy
            """)
    
    # Legal disclaimer
    st.markdown("---")
    st.warning("""
    ⚖️ **Legal Disclaimer:** This application is for educational and awareness purposes only. 
    The results displayed do not replace comprehensive medical examination and specialized medical consultation. 
    Always consult a qualified doctor before making any health decisions.
    """)

if __name__ == "__main__":
    main()