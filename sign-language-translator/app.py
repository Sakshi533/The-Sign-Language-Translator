# # app.py
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras
import os

# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'sign_language_model_v2.h5')

# Set page config
st.set_page_config(
    page_title='SignBridge: Instant Sign Language Communication',
    page_icon=':hand:',
    layout='centered'
)

# Load the model first
try:
    model = keras.models.load_model(model_path)
    input_shape = model.input_shape[1:]
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.error(f"Model path: {model_path}")
    st.stop()

# Title and description
st.title("SignBridge 🖐️ - ASL to Text Translator")
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <p style='font-size: 18px;'>
           Upload a hand gesture image, and watch SignBridge translate it into English letters 
            using real-time AI prediction.
        </p>
    </div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    # Instructions
    st.subheader('Instructions')
    st.markdown("""
1. Upload a photo showing a clear ASL sign using one hand  
2. Make sure the background is plain and the hand is visible  
3. The app will predict the corresponding English alphabet letter  
4. Review the uploaded image and model prediction  
""")

    # Add file uploader
    uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

with col2:
    # ASL alphabet mapping (excluding J and Z as they require motion)
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    if uploaded_file is not None:
        try:
            # Process the image
            image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
            img = image.resize(input_shape[:2])  # Resize to match model input
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class] * 100

            # Display results
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h2>Prediction: {alphabet[predicted_class]}</h2>
                    <p style='font-size: 16px;'>Confidence: {confidence:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

            # Display the uploaded image
            st.image(image, caption='Uploaded Image', use_column_width=True)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Add footer
st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <p style='color: gray; font-size: 14px;'>
            Note: This model recognizes static ASL letters (A-Y, excluding J and Z which require motion).
        </p>
    </div>
""", unsafe_allow_html=True)
#python -m streamlit run "d:/sign language githuvb/sign-language-translator/app.py"
