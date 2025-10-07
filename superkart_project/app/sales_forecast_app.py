import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# ============================================
# 🌟 App Configuration
# ============================================
st.set_page_config(
    page_title="📈 Sales Forecast Failure Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Sales Forecast Failure Prediction App")
st.markdown("""
Welcome! This app predicts whether your **sales forecast is likely to fail or succeed** based on the key business inputs you provide.
The model was **trained using historical sales data** and deployed via Hugging Face 🤗.
""")

# ============================================
# 🔁 Load Pre-trained Model from Hugging Face
# ============================================

# 👉 Replace this with your Hugging Face username
HF_USER_ID = "vikas0615"
MODEL_REPO_ID = f"{HF_USER_ID}/Sales-Forecast-Model"
MODEL_FILENAME = "xgb_best_model.joblib"

@st.cache_resource
def load_model_from_huggingface():
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error("❌ Could not download or load the model. Please check the Hugging Face repo details.")
        st.stop()

model = load_model_from_huggingface()
st.success("✅ Pre-trained model successfully loaded from Hugging Face Hub.")

# ============================================
# 📥 Collect User Inputs
# ============================================

st.subheader("📊 Enter Sales Forecast Details")

col1, col2 = st.columns(2)
with col1:
    historical_sales = st.number_input("📦 Historical Average Sales (Units)", min_value=0, value=500)
    marketing_spend = st.number_input("💰 Marketing Spend ($)", min_value=0.0, value=2000.0)
    promotion_discount = st.number_input("🏷️ Promotion Discount (%)", min_value=0.0, max_value=100.0, value=10.0)

with col2:
    competitor_price = st.number_input("💼 Competitor Avg. Price ($)", min_value=0.0, value=50.0)
    seasonal_index = st.slider("📅 Seasonal Demand Index (0-1)", 0.0, 1.0, 0.5)
    holiday_flag = st.selectbox("🎉 Is there a Holiday Season?", ["No", "Yes"])

# Convert categorical input to numeric
holiday_flag_val = 1 if holiday_flag == "Yes" else 0

# Prepare input DataFrame
input_data = pd.DataFrame([{
    "historical_sales": historical_sales,
    "marketing_spend": marketing_spend,
    "promotion_discount": promotion_discount,
    "competitor_price": competitor_price,
    "seasonal_index": seasonal_index,
    "holiday_flag": holiday_flag_val
}])

st.write("📋 **Input Summary:**")
st.dataframe(input_data)

# ============================================
# 🔮 Prediction
# ============================================

if st.button("🔍 Predict Forecast Outcome"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # probability of failure

        if prediction == 1:
            st.error(f"⚠️ Prediction: Sales Forecast is **likely to FAIL**. (Confidence: {probability*100:.2f}%)")
            st.progress(int(probability * 100))
        else:
            st.success(f"✅ Prediction: Sales Forecast is **likely to SUCCEED**. (Confidence: {(1-probability)*100:.2f}%)")
            st.progress(int((1 - probability) * 100))

    except Exception as e:
        st.error(f"❌ Prediction failed. Error: {e}")

# ============================================
# 📊 Footer Section
# ============================================
st.markdown("---")
st.markdown("""
💡 **Tip:** This prediction is based on historical data and business factors.
Use it to **complement your decision-making**, not replace it.
""")
