import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# Load model & scaler
scaler = joblib.load("scaler.pkl")
model = load_model("salary_prediction_model.h5")

# Sidebar
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Navigate", ["Home", "Predict Salary", "About"])

# ---------- HOME ----------
if page == "Home":
    st.markdown("<h1 style='text-align:center;'>💼 Salary Prediction App</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align:center; font-size:18px;'>
        Predict salary based on <b>Years of Experience</b><br>
        using Machine Learning
        </p>
        """,
        unsafe_allow_html=True
    )
    st.success("👉 Use the sidebar to predict salary")

# ---------- PREDICT ----------
elif page == "Predict Salary":
    st.title("📊 Predict Your Salary")

    years = st.slider("Years of Experience", 0, 20, 2)

    if st.button("Predict Salary 💰"):
        result = model.predict(np.array([[years]]))
        salary = scaler.inverse_transform(result)[0][0]

        st.success(f"💵 Estimated Salary: ₹ {salary:,.2f}")

        # Graph
        st.subheader("📈 Salary Trend")
        x = np.arange(0, 21)
        y = scaler.inverse_transform(model.predict(x.reshape(-1, 1)))

        fig, ax = plt.subplots()
        ax.plot(x, y, marker="o")
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.set_title("Salary vs Experience")

        st.pyplot(fig)

# ---------- ABOUT ----------
elif page == "About":
    st.title("ℹ️ About This App")
    st.write("""
    **Salary Prediction App**

    - Built with **Streamlit**
    - Uses **Machine Learning**
    - Input: Years of Experience
    - Output: Predicted Salary

    🎓 Mini Project / Demo App
    """)
