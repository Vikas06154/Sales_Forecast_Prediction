import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="vikas0615/Sales-Forecast-Model", filename="best_sales_forecast_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Sales Forecast Prediction
st.title("Sales Forecast Prediction App")
st.write("""
This application predicts the likelihood of sales based on its operational parameters.
Please enter the data below to get a prediction.
""")

# User input
Store_Id = st.selectbox("Store_Id", ["OUT001", "OUT002", "OUT003", OUT004])
Store_Size = st.selectbox("Store_Id", ["High", "Medium", "Small"])
Store_Location_City_Type = st.selectbox("Store_Location_City_Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store_Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
Product_Sugar_Content = st.selectbox("Product_Sugar_Content", ["Low Sugar", "Regular", "No Sugar"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Store_Id': Store_Id,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type,
    'Product_Sugar_Content': Product_Sugar_Content,
}])


if st.button("Sales Forecast Prediction"):
    prediction = model.predict(input_data)[0]
    result = "Sales Forecast Prediction" if prediction == 1 else "No Forecast"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
