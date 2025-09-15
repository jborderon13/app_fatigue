import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Charger le modèle
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = joblib.load(f)
    return model

model = load_model()

# Labels des classes (cycles avant rupture)
labels = [r"[0, $10^5$]", r"[$10^5$, $10^7$]", r"[$10^7$, +∞]"]

# Configuration de la page
st.set_page_config(
    page_title="Classification des Sols - Cycles avant rupture",
    page_icon="🌍",
    layout="wide"
)

# Titre et description
st.title("🌍 Application de Classification des Sols")
st.markdown("""
Cette application prédit la catégorie de cycles avant rupture d'un sol en fonction de ses caractéristiques.
Remplissez les champs ci-dessous et cliquez sur **Prédire** pour obtenir le résultat et les probabilités associées.
""")

# Séparateur visuel
st.markdown("---")

# Organisation des champs en colonnes
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

# Bouton de prédiction centré
st.markdown("---")
if st.button("🔮 **Prédire la catégorie du sol**", type="primary", use_container_width=True):
    # Préparation des données (avec tes noms de colonnes exacts)
    data = {
        'Simplified USCS': simplified_uscs,
        'Gravel content (%)': gravel_content,
        'Sand content (%) ': sand_content,  # Espace supplémentaire conservé
        'Fine particles content (%)': fine_particles_content,
        'Plasticity index ': plasticity_index,  # Espace supplémentaire conservé
        'Liquid limit (%) ': liquid_limit,  # Espace supplémentaire conservé
        'Plastic limit (%)': plastic_limit,
        'Cement content (%)': cement_content,
        'Cement classification': cement_classification,
        'Lime content (%)': lime_content,
        'Curing duration (days)': curing_duration,
        'Curing temperature (°C)': curing_temperature,
        'Density (g/cm^3)': density,  # Nom exact avec "^3"
        'Water content (%)': water_content,
        'Frequency (Hz)': frequency,
        'SR (Stress Ratio) (-)': sr
    }

    input_df = pd.DataFrame(data, index=[0])
    input_data = input_df.fillna(0)

    # Prédiction et probabilités
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    # Affichage du résultat
    st.markdown("---")
    st.subheader("📊 Résultat de la prédiction")
    st.success(f"La catégorie prédite est : **{labels[prediction[0]]}**", icon="✅")

    # Affichage des probabilités
    st.subheader("📈 Probabilités d'appartenance à chaque catégorie de cycles avant rupture")
    prob_df = pd.DataFrame({
        "Catégorie (cycles avant rupture)": labels,
        "Probabilité": probabilities[0]
    })
    st.dataframe(prob_df.style.format({"Probabilité": "{:.2%}"}))

    # Graphique en barres des probabilités
    fig, ax = plt.subplots()
    ax.bar(labels, probabilities[0], color=['#4e79a7', '#f28e2b', '#e15759'])
    ax.set_ylabel("Probabilité")
    ax.set_title("Probabilités d'appartenance aux catégories")
    st.pyplot(fig)

    # Affichage des paramètres saisis
    st.subheader("Paramètres saisis")
    st.dataframe(input_df.style.highlight_max(axis=0))
