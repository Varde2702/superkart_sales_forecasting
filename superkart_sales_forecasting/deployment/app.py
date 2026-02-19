import streamlit as st
import pandas as pd
import joblib

model = joblib.load('best_model.pkl')

st.title('SuperKart Sales Forecasting App')
file = st.file_uploader('Upload CSV File')

if file:
    df = pd.read_csv(file)
    preds = model.predict(df)
    df['Predicted_Sales'] = preds
    st.write(df)
