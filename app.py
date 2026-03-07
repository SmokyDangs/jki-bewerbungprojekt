import streamlit as st

from ultralytics import YOLO

from PIL import Image

import numpy as np

import pandas as pd



# Konfiguration

st.set_page_config(

    page_title="JKI Crop & Pest Detector",

    page_icon="🌱",

    layout="wide"

)



# Custom CSS für JKI-Branding

st.markdown("""

    <style>

    .main { background-color: #f9fbf9; }

    [data-testid="stSidebar"] { background-color: #f0f4f0; }

    .stMetric { background-color: #ffffff; border: 1px solid #e1e4e8; padding: 10px; border-radius: 8px; margin-bottom: 10px; }

    </style>

    """, unsafe_allow_html=True)



# Mapping-Dictionary für die deutschen Übersetzungen

DEUTSCHE_NAMEN = {

    "Adulto": "Erwachsenes Tier",

    "Black-grass-caterpillar": "Schwarze Graseule (Raupe)",

    "Cerambycidae_larvae": "Bockkäfer-Larve",

    "Cnidocampa_flavescens-walker_pupa-": "Gelbe Schlupfwespe (Puppe)",

    "Coconut-black-headed-caterpillar": "Kokospalmen-Raupe",

    "Cricket": "Grille",

    "Diamondback-moth": "Kohlmotte",

    "Drosicha_contrahens_female": "Schildlaus (Weibchen)",

    "Erthesina_fullo_nymph-2": "Gelbgefleckte Baumwanze (Nymphe)",

    "Grasshopper": "Heuschrecke",

    "Hyphantria_cunea_larvae": "Amerikanischer Webebär (Larve)",

    "Hyphantria_cunea_pupa": "Amerikanischer Webebär (Puppe)",

    "Latoia_consocia_walker_larvae": "Schneckenspinner-Larve",

    "Leaf-eating-caterpillar": "Blattfressende Raupe",

    "Micromelalopha_troglodyta-graeser-_larvae": "Pappelspinner-Larve",

    "Ninfa": "Nymphe",

    "Ovo": "Ei",

    "Phenacoccus": "Schmierlaus",

    "Plagiodera_versicolora-laicharting-": "Weidenblattkäfer",

    "Plagiodera_versicolora-laicharting-_larvae": "Weidenblattkäfer (Larve)",

    "Plagiodera_versicolora-laicharting-_ovum": "Weidenblattkäfer (Ei)",

    "Red cotton steiner": "Rote Baumwollwanze",

    "Sericinus_montela_larvae": "Schwalbenschwanz-Raupe",

    "Spilarctia_subcarnea-walker-_larvae-2": "Bärenspinner-Raupe",

    "Wereng coklat": "Braune Zikade",

    "Aphid": "Blattlaus",

    "Citricola scale": "Zitrus-Schildlaus"

}



# Modell laden

@st.cache_resource

def load_model():

    return YOLO("best.pt")



model = load_model()



# --- SIDEBAR ---

with st.sidebar:

    st.image("https://www.julius-kuehn.de/fileadmin/templates/jki/img/jki_logo.png", width=120)

    st.header("Analyse & Befunde")

    

    conf_threshold = st.slider("KI-Konfidenz", 0.0, 1.0, 0.25, 0.05, 

                               help="Schwellenwert für die Erkennungssicherheit.")

    

    st.divider()

    

    # Platzhalter für Befunde in der Sidebar

    sidebar_results_placeholder = st.container()



# --- HAUPTBEREICH ---

st.title("🌱 JKI Prototyp: Crop & Weed Detector")

st.markdown("#### KI-gestützte Schaderreger-Diagnostik")



uploaded_file = st.file_uploader("Pflanzenfoto zur Analyse hochladen...", type=["jpg", "jpeg", "png"])



if uploaded_file:

    with st.spinner('Bild wird analysiert...'):

        image = Image.open(uploaded_file)

        

        # Inferenz

        results = model.predict(image, conf=conf_threshold)

        res_plotted = results[0].plot()

        

        # Anzeige der Bilder im Hauptbereich

        col1, col2 = st.columns(2)

        with col1:

            st.image(image, caption="Originalaufnahme", use_container_width=True)

        with col2:

            st.image(res_plotted, caption="Visualisierte Erkennung", use_container_width=True)



        # Daten für die Sidebar aufbereiten

        detections = results[0].boxes.cls.tolist()

        names = model.names

        found_any = False

        
