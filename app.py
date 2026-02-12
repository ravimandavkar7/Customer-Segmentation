import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🧠 Customer Segmentation App")

st.write("Enter customer details to find segment")

# Input fields
income = st.number_input("Annual Income")
spending = st.number_input("Spending Score (1-100)")

if st.button("Predict Segment"):
    data = np.array([[income, spending]])
    data_scaled = scaler.transform(data)
    cluster = model.predict(data_scaled)
    
    labels = {
    0: "Low Income - Low Spending",
    1: "High Value Customer",
    2: "Medium Customer",
    3: "Low Value Customer"
     }

    st.success(f"Customer Segment: {labels[cluster[0]]}")
