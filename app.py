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

# Custom CSS für JKI-Look (Dezentes Grün)
st.markdown("""
    <style>
    .main { background-color: #f9fbf9; }
    .stMetric { background-color: #ffffff; border: 1px solid #e1e4e8; padding: 15px; border-radius: 10px; }
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

# Sidebar für Experten-Einstellungen
with st.sidebar:
    st.image("https://www.julius-kuehn.de/fileadmin/templates/jki/img/jki_logo.png", width=150) # Platzhalter für JKI Logo
    st.header("Analyse-Einstellungen")
    conf_threshold = st.slider("KI-Konfidenz-Schwellenwert", 0.0, 1.0, 0.25, 0.05, 
                               help="Bestimmt, ab welcher Wahrscheinlichkeit ein Objekt als erkannt gilt.")
    st.info("Dieses Tool dient zur automatisierten Früherkennung von Schädlingen in landwirtschaftlichen Kulturen.")

# Hauptbereich
st.title("🌱 JKI Prototyp: Crop & Weed Detector")
st.markdown("### Präzisionslandwirtschaft & Pflanzenschutz-Monitoring")
st.divider()

uploaded_file = st.file_uploader("Bild zur Analyse hochladen (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Ladeanzeige
    with st.spinner('Bild wird verarbeitet...'):
        image = Image.open(uploaded_file)
        
        # Inferenz mit einstellbarem Schwellenwert
        results = model.predict(image, conf=conf_threshold)
        
        # Ergebnisse visualisieren
        res_plotted = results[0].plot()
        
        # Layout Spalten
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Originalaufnahme", use_container_width=True)
        with col2:
            st.image(res_plotted, caption="Visualisierte Erkennung (KI)", use_container_width=True)

        st.divider()
        
        # Statistik & Auswertung
        st.subheader("Befundbericht")
        
        detections = results[0].boxes.cls.tolist()
        names = model.names
        
        # Daten für Tabelle und Metriken sammeln
        found_data = []
        
        # Metriken in Reihen anzeigen
        metric_cols = st.columns(4)
        m_idx = 0
        
        for idx, name in names.items():
            count = detections.count(idx)
            if count > 0:
                name_de = DEUTSCHE_NAMEN.get(name, name)
                
                # Metrik-Karten (max 4 pro Zeile)
                with metric_cols[m_idx % 4]:
                    st.metric(label=name_de, value=int(count))
                m_idx += 1
                
                found_data.append({"Schädling/Stadium": name_de, "Anzahl": int(count)})

        # Zusätzliche Details in Tabelle
        if found_data:
            with st.expander("Detaillierte Tabellenansicht"):
                df = pd.DataFrame(found_data)
                st.table(df)
        else:
            st.warning(f"Keine Objekte mit einer Konfidenz von > {conf_threshold*100}% gefunden. Versuchen Sie den Schwellenwert in der Sidebar zu senken.")

else:
    # Willkommensbildschirm / Platzhalter
    st.info("Bitte laden Sie ein Foto hoch, um die KI-gestützte Analyse zu starten.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Corn_field_in_summer_Germany.jpg/1200px-Corn_field_in_summer_Germany.jpg", caption="Beispielhafte Feldaufnahme", use_container_width=True, alpha=0.3)

# Footer
st.markdown("---")
st.caption("Entwickelt als Prototyp für das Julius Kühn-Institut (JKI) | Fachbereich: Intelligente Pflanzenschutzsysteme")
