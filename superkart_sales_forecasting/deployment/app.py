import streamlit as st
import pandas as pd
import joblib
import os

# Try loading from deployment folder or local directory
model_path = 'deployment/best_model.pkl' if os.path.exists('deployment/best_model.pkl') else 'best_model.pkl'

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f'Error loading model: {e}')

st.title('SuperKart Sales Forecasting App')
file = st.file_uploader('Upload CSV File')

if file:
    df = pd.read_csv(file)
    preds = model.predict(df)
    df['Predicted_Sales'] = preds
    st.write(df)
