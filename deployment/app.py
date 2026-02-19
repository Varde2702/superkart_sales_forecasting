import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# Force redeploy: Updated at 2024-02-19
model_path = 'best_model.pkl' if os.path.exists('best_model.pkl') else 'deployment/best_model.pkl'

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f'Error loading model: {e}')

st.title('SuperKart Sales Forecasting App')
st.write(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
file = st.file_uploader('Upload CSV File')

if file:
    df = pd.read_csv(file)
    preds = model.predict(df)
    df['Predicted_Sales'] = preds
    st.write(df)
