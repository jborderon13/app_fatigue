import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Logo
st.image("logo.jpg", width=1050)  # Adjust width as needed

# Load the model
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = joblib.load(f)
    return model

model = load_model()

# Class labels (cycles to failure)
labels = [r"[0, $10^5$]", r"[$10^5$, $10^7$]", r"[$10^7$, +∞]"]

# Page configuration
st.set_page_config(
    page_title="Soil Classification - Cycles to Failure",
    page_icon="🌍",
    layout="wide"
)

# Title and description
st.title("🌍 Soil Classification Application")
st.markdown("""
This application predicts the **cycles-to-failure category** of a soil sample based on its properties.  
Fill in the fields below and click **Predict** to see the result and the associated probabilities.
""")

# Visual separator
st.markdown("---")

# Organize input fields into columns
col1, col2 = st.columns(2)

with col1:
    simplified_uscs = st.text_input('**Simplified USCS**', value="SP")
    gravel_content = st.number_input('**Gravel content (%)**', value=0.0, min_value=0.0, max_value=100.0)
    sand_content = st.number_input('**Sand content (%)**', value=80.0, min_value=0.0, max_value=100.0)
    fine_particles_content = st.number_input('**Fine particles content (%)**', value=20.0, min_value=0.0, max_value=100.0)
    plasticity_index = st.number_input('**Plasticity index**', value=0.0, min_value=0.0)
    liquid_limit = st.number_input('**Liquid limit (%)**', value=0.0, min_value=0.0)
    plastic_limit = st.number_input('**Plastic limit (%)**', value=0.0, min_value=0.0)
    cement_content = st.number_input('**Cement content (%)**', value=0.0, min_value=0.0, max_value=100.0)

with col2:
    cement_classification = st.text_input('**Cement classification**', value="CEM I")
    lime_content = st.number_input('**Lime content (%)**', value=0.0, min_value=0.0, max_value=100.0)
    curing_duration = st.number_input('**Curing duration (days)**', value=7.0, min_value=0.0)
    curing_temperature = st.number_input('**Curing temperature (°C)**', value=20.0, min_value=-20.0, max_value=100.0)
    density = st.number_input('**Density (g/cm³)**', value=2.0, min_value=0.1, max_value=10.0)
    water_content = st.number_input('**Water content (%)**', value=10.0, min_value=0.0, max_value=100.0)
    frequency = st.number_input('**Frequency (Hz)**', value=0.0, min_value=0.0)
    sr = st.number_input('**SR (Stress Ratio) (-)**', value=0.0)

# Prediction button
st.markdown("---")
if st.button("🔮 **Predict Soil Category**", type="primary", use_container_width=True):
    # Prepare input data
    data = {
        'Simplified USCS': simplified_uscs,
        'Gravel content (%)': gravel_content,
        'Sand content (%) ': sand_content,  # Keep exact column name if trained this way
        'Fine particles content (%)': fine_particles_content,
        'Plasticity index ': plasticity_index,  # Keep exact column name
        'Liquid limit (%) ': liquid_limit,      # Keep exact column name
        'Plastic limit (%)': plastic_limit,
        'Cement content (%)': cement_content,
        'Cement classification': cement_classification,
        'Lime content (%)': lime_content,
        'Curing duration (days)': curing_duration,
        'Curing temperature (°C)': curing_temperature,
        'Density (g/cm^3)': density,  # Exact column name
        'Water content (%)': water_content,
        'Frequency (Hz)': frequency,
        'SR (Stress Ratio) (-)': sr
    }

    input_df = pd.DataFrame(data, index=[0])
    input_data = input_df.fillna(0)

    # Prediction and probabilities
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    # Display result
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.success(f"The predicted category is: **{labels[prediction[0]]}**", icon="✅")

    # Display probabilities
    st.subheader("📈 Probabilities of belonging to each cycles-to-failure category")
    prob_df = pd.DataFrame({
        "Category (cycles to failure)": labels,
        "Probability": probabilities[0]
    })
    st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

    # Probability bar chart
    fig, ax = plt.subplots()
    ax.bar(labels, probabilities[0], color=['#4e79a7', '#f28e2b', '#e15759'])
    ax.set_ylabel("Probability")
    ax.set_title("Category Probabilities")
    st.pyplot(fig)

    # Display entered parameters
    st.subheader("Entered Parameters")
    st.dataframe(input_df.style.highlight_max(axis=0))
