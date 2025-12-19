import streamlit as st
import numpy as np
st.title("Salary Prediction App")
import joblib
from tensorflow.keras.models import load_model
scaler=joblib.load('scaler.pkl')
model=load_model('salary_prediction_model.h5')
n=st.slider("Enter Years of Experience",1,15,2)
if st.button("Predict Salary"):
    result=model.predict(np.array([[n]]))
    result=scaler.inverse_transform(result)
    st.write("The Salary is:",result[0][0])