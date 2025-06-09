import streamlit as st
import joblib
import pandas as pd

# Page title and instructions
st.set_page_config(page_title="Issue Classifier", page_icon="🔧", layout="wide")
st.title("🔧 Issue Category Classifier")
st.markdown("""
**Instructions:**  
Enter a description of your hardware or software issue in the text box below.  
Click the "Predict Category" button to find out what category your issue belongs to.
""")

# Sidebar with examples and info
with st.sidebar:
    st.header("Examples")
    st.markdown("""
    - **My laptop screen is cracked**  
    - **Battery drains too fast**  
    - **WiFi keeps disconnecting**  
    - **Laptop overheating a lot**  
    - **System very slow while opening apps**  
    """)
    st.markdown("---")
    st.markdown("**Note:** This tool uses a machine learning model to predict the category of your issue.")

# Load the trained model
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error("Model file not found or error loading model. Please ensure 'model.pkl' is in your directory.")
    st.stop()

# User input
user_input = st.text_area(
    "Describe your issue here:",
    placeholder="e.g., My laptop overheats and turns off",
    height=100
)

# Prediction button and result
col1, col2 = st.columns([1, 2])
with col1:
    predict_button = st.button("Predict Category", type="primary", use_container_width=True)

if predict_button and user_input.strip() != "":
    with st.spinner("Analyzing your issue..."):
        prediction = model.predict([user_input])[0]
    st.success(f"**Predicted Category:** {prediction}")
elif predict_button and user_input.strip() == "":
    st.warning("Please enter a description of your issue.")

# Clear button
if st.button("Clear Input", type="secondary"):
    user_input = ""
    # Note: Streamlit does not support direct clearing of st.text_area in this way.
    # For a true reset, the page will refresh.
    st.rerun()

# Footer
st.markdown("---")
st.caption("© 2024 Issue Classifier. Powered by Streamlit and scikit-learn.")
