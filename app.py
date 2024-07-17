import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model from disk
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Feature names and target names
feature_names = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
target_names = ['setosa', 'versicolor', 'virginica']

# Streamlit app
st.set_page_config(page_title="Iris Species Prediction", page_icon="🌸", layout="centered")

st.title("🌸 Iris Species Prediction")
st.markdown("""
<style>
    .main {
        background-color: #f0f0f5;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stSlider {
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

st.write("Enter the features of the Iris flower to predict its species:")

# Input sliders for features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Prepare input data for prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict the species
if st.button("Predict"):
    prediction = model.predict(input_data)
    predicted_species = target_names[prediction[0]]
    st.success(f"The predicted species is: **{predicted_species}**")

st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h4>Created by Harman</h4>
        <p>This Streamlit app predicts the species of an Iris flower based on its features.</p>
    </div>
""", unsafe_allow_html=True)
