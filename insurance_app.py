import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
# Inject Custom CSS to turn the sidebar background red
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #D32F2F !important;
    }
    /* This makes all text inside the sidebar white so it stays readable */
    [data-testid="stSidebar"] __element__ , [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] a {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 1. Page Configuration
st.set_page_config(page_title="🏥 Insurance Predictor", page_icon="🏥", layout="wide")

# 2. Load the trained components safely
@st.cache_resource
def load_ml_assets():
    model_path = "insurance_model.pkl"
    scaler_path = "insurance_scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

model, scaler = load_ml_assets()

# 3. Sidebar - Profile & Social Accounts ONLY
with st.sidebar:
    st.header("💻 Application Developer")
    if os.path.exists("IMG-20260704-WA0629.jpg"):
        st.image("IMG-20260704-WA0629.jpg", caption="Sulayman Bah mechine and deep learning enginner", width=180)
    else:
        st.info("💡 Profile image asset not found.")
        
    st.markdown("---")
    st.subheader("🔗 Connect With Me")
    
    # Update these URLs with your real profile links
    st.markdown("[📁 GitHub Profile](https://github.com)")
    st.markdown("[💼 LinkedIn Profile](https://linkedin.com/in/WIN_20250906_05_26_12_Pro.jpg)")
    st.markdown("[📧 Email Support](http://bahsulayman689@gmail.com)")

# 4. Main Page Header Layout
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.title("🏥 Medical Insurance Cost Estimator")
    st.write("Fill out the demographic form below to estimate annual medical insurance charges.")
    st.write("""
    Hi! I'm Sulayman Bah.
    I'm a mechine learning and deep learning enginner.
    I build Machine Learning and
    Deep Learning applications
    using Python and Streamlit.
    """)
with header_col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=200)
    else:
        st.info("💡 Logo asset not found.")
    
st.markdown("---")

# 5. Missing Files Warning
if model is None or scaler is None:
    st.error("🚨 Error: 'insurance_model.pkl' or 'insurance_scaler.pkl' was not found in this folder!")
    st.info("Please make sure you have run your training script and that the saved model files are inside the exact same folder as this app.py file.")
else:
    # 6. Center Input Form (Main Page Grid)
    st.subheader("📋 Enter Personal Metrics")
    
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        age = st.number_input("Age", min_value=1, max_value=120,value=30, step=1)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
        
    with input_col2:
        sex_input = st.selectbox("Biological Sex", options=["Male", "Female"])
        smoker_input = st.selectbox("Do you smoke?", options=["No", "Yes"])
        
    st.markdown("---")
    
    # 7. Action Predict Button on Main Page
    if st.button("🔮 Calculate Predicted Charges", type="primary", use_container_width=True):
        
        # Map text inputs back to numerical values
        sex = 1 if sex_input == "Male" else 0
        smoker = 1 if smoker_input == "Yes" else 0

        # Structure data exactly as it was during training
        feature_names = ["age", "sex", "bmi", "children", "smoker"]
        input_data = pd.DataFrame([[age, sex, bmi, children, smoker]], columns=feature_names)
        
        try:
            # Transform and Predict
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)
            
            # Format outputs safely
            final_charge = max(0.0, float(prediction.item()))

            # 8. UI Dashboard Panels for Displaying Outputs
            result_col1, result_col2, result_col3 = st.columns(3)
            st.subheader("📊 Price Breakdown Results")
            with result_col1:
                st.metric(label="Estimated Annual Premium", value=f"${final_charge:,.2f}")
            with result_col2:
                # Calculate the exact monthly installment
                monthly_charge = final_charge / 12
                st.metric(label="Estimated Monthly Payment", value=f"${monthly_charge:,.2f}")
                
            with result_col3:
                if smoker_input == "Yes":
                    st.warning("⚠️ High Risk Tier (Smoker)")
                    st.info("You need to stop smoking ")
                    st.snow()
                else:
                    st.success("✅ Standard Risk Tier (Non-Smoker)")
                    st.info("You are good to sir and madam")
                    st.balloons()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
with st.sidebar.expander("👤 About Me"):

    st.write("""
    Hi! I'm Sulayman Bah.
    I'm a mechine learning and deep learning enginner.
    I build Machine Learning and
    Deep Learning applications
    using Python and Streamlit.
    """)
st.sidebar.subheader("🛠 Skills")

st.sidebar.write("🐍 Python")
st.sidebar.write("🤖 Machine Learning")
st.sidebar.write("📊 Data Analysis")
st.sidebar.write("🎨 Deep learning")
st.sidebar.write("🧠 Software enginner")
st.divider()
with st.sidebar:
    st.image("logo.png", width=150)
    st.title("ML Ops Dashboard")
    st.markdown("---")