import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Charger le modèle
with open("best_model.pkl", "rb") as f:
    model = joblib.load(f)

# Titre de l'application
st.title("Application de Classification des Sols")

# Description
st.write("""
Cette application permet de prédire la catégorie d'un sol en fonction de ses caractéristiques.
Remplis les champs ci-dessous et clique sur le bouton **Prédire** pour obtenir le résultat.
""")

# Fonction pour créer les champs de saisie
def user_input_features():
    st.sidebar.header('Paramètres du Sol')

    # Initialisation des valeurs par défaut (peut être modifié)
    default_values = {
        'Simplified USCS': 'SP',
        'Gravel content (%)': 0.0,
        'Sand content (%)': 80.0,
        'Fine particles content (%)': 20.0,
        'Plasticity index': 0.0,
        'Liquid limit (%)': 0.0,
        'Plastic limit (%)': 0.0,
        'Cement content (%)': 0.0,
        'Cement classification': 'CEM I',
        'Lime content (%)': 0.0,
        'Curing duration (days)': 7.0,
        'Curing temperature (°C)': 20.0,
        'Density (g/cm³)': 2.0,
        'Water content (%)': 10.0,
        'Frequency (Hz)': 0.0,
        'SR (Stress Ratio) (-)': 0.0
    }

    # Création des champs de saisie
    simplified_uscs = st.sidebar.text_input('Simplified USCS', default_values['Simplified USCS'])
    gravel_content = st.sidebar.number_input('Gravel content (%)', value=default_values['Gravel content (%)'])
    sand_content = st.sidebar.number_input('Sand content (%)', value=default_values['Sand content (%)'])
    fine_particles_content = st.sidebar.number_input('Fine particles content (%)', value=default_values['Fine particles content (%)'])
    plasticity_index = st.sidebar.number_input('Plasticity index', value=default_values['Plasticity index'])
    liquid_limit = st.sidebar.number_input('Liquid limit (%)', value=default_values['Liquid limit (%)'])
    plastic_limit = st.sidebar.number_input('Plastic limit (%)', value=default_values['Plastic limit (%)'])
    cement_content = st.sidebar.number_input('Cement content (%)', value=default_values['Cement content (%)'])
    cement_classification = st.sidebar.text_input('Cement classification', default_values['Cement classification'])
    lime_content = st.sidebar.number_input('Lime content (%)', value=default_values['Lime content (%)'])
    curing_duration = st.sidebar.number_input('Curing duration (days)', value=default_values['Curing duration (days)'])
    curing_temperature = st.sidebar.number_input('Curing temperature (°C)', value=default_values['Curing temperature (°C)'])
    density = st.sidebar.number_input('Density (g/cm³)', value=default_values['Density (g/cm³)'])
    water_content = st.sidebar.number_input('Water content (%)', value=default_values['Water content (%)'])
    frequency = st.sidebar.number_input('Frequency (Hz)', value=default_values['Frequency (Hz)'])
    sr = st.sidebar.number_input('SR (Stress Ratio) (-)', value=default_values['SR (Stress Ratio) (-)'])

    # Création d'un dictionnaire avec les valeurs saisies
    data = {
        'Simplified USCS': simplified_uscs,
        'Gravel content (%)': gravel_content,
        'Sand content (%)': sand_content,
        'Fine particles content (%)': fine_particles_content,
        'Plasticity index': plasticity_index,
        'Liquid limit (%)': liquid_limit,
        'Plastic limit (%)': plastic_limit,
        'Cement content (%)': cement_content,
        'Cement classification': cement_classification,
        'Lime content (%)': lime_content,
        'Curing duration (days)': curing_duration,
        'Curing temperature (°C)': curing_temperature,
        'Density (g/cm³)': density,
        'Water content (%)': water_content,
        'Frequency (Hz)': frequency,
        'SR (Stress Ratio) (-)': sr
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Récupération des données saisies par l'utilisateur
input_df = user_input_features()

# Affichage des données saisies
st.subheader('Paramètres saisis')
st.write(input_df)

# Bouton de prédiction
if st.button('Prédire'):
    # Préparation des données pour la prédiction (gérer les NaN si nécessaire)
    input_data = input_df.copy()
    # Remplace les valeurs vides ou NaN par 0 ou une autre valeur par défaut, selon ton modèle
    input_data = input_data.fillna(0)

    # Prédiction
    prediction = model.predict(input_data)

    # Affichage du résultat
    st.subheader('Résultat de la prédiction')
    st.write(f"La catégorie prédite est : **{prediction[0]}**")
