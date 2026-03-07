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
    .stExpander { background-color: #ffffff; }
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
    model = YOLO("best.pt")
    if hasattr(model, 'names'):
        for idx, eng_name in model.names.items():
            if eng_name in DEUTSCHE_NAMEN:
                model.names[idx] = DEUTSCHE_NAMEN[eng_name]
    return model

model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.julius-kuehn.de/fileadmin/templates/jki/img/jki_logo.png", width=120)
    st.header("Analyse-Optionen")
    
    conf_threshold = st.slider("KI-Konfidenz", 0.0, 1.0, 0.25, 0.05, 
                               help="Schwellenwert für die Erkennungssicherheit.")
    
    st.divider()
    st.subheader("Gesamtbefund (Batch)")
    sidebar_batch_placeholder = st.empty()

# --- HAUPTBEREICH ---
st.title("🌱 JKI Prototyp: Crop & Weed Detector")
st.markdown("#### KI-gestützte Schaderreger-Diagnostik (Einzel- & Batchverarbeitung)")

uploaded_files = st.file_uploader(
    "Pflanzenfotos hochladen (einzeln oder mehrere)...", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    all_detections = []
    
    # Fortschrittsbalken für Batch
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Ergebnisse anzeigen
    st.subheader(f"Verarbeitung von {len(uploaded_files)} Bild(ern)")
    
    for i, file in enumerate(uploaded_files):
        # Fortschritt aktualisieren
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Analysiere Bild {i+1} von {len(uploaded_files)}: {file.name}")
        
        image = Image.open(file)
        
        # Inferenz
        results = model.predict(image, conf=conf_threshold)
        res_plotted = results[0].plot()
        
        # Jedes Bild in einem Expander anzeigen, um Platz zu sparen
        with st.expander(f"Ergebnis: {file.name}", expanded=(len(uploaded_files) == 1)):
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(res_plotted, caption="Detektion (Deutsch)", use_container_width=True)
            
            # Einzelbefunde pro Bild
            current_detections = results[0].boxes.cls.tolist()
            all_detections.extend(current_detections)
            
            if current_detections:
                found_names = [model.names[int(cls)] for cls in current_detections]
                st.write(f"**Gefunden:** {', '.join(set(found_names))} ({len(current_detections)} Objekte)")
            else:
                st.write("Keine Schädlinge gefunden.")

    # Status aufräumen
    status_text.success(f"Analyse von {len(uploaded_files)} Bildern abgeschlossen.")
    
    # --- BATCH-AUSWERTUNG IN DER SIDEBAR ---
    with sidebar_batch_placeholder.container():
        if all_detections:
            unique_ids = sorted(list(set(all_detections)))
            for idx in unique_ids:
                total_count = all_detections.count(idx)
                st.metric(label=model.names[idx], value=int(total_count))
            st.info(f"Gesamtanzahl Funde: {len(all_detections)}")
        else:
            st.warning("Keine Befunde im Batch.")

else:
    st.info("Bitte laden Sie ein oder mehrere Fotos hoch.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Corn_field_in_summer_Germany.jpg/1200px-Corn_field_in_summer_Germany.jpg", 
             caption="Überwachung landwirtschaftlicher Kulturen", 
             use_container_width=True)

# Footer
st.markdown("---")
st.caption("Prototyp für das Julius Kühn-Institut (JKI) | Monitoring-System v1.3 - Batch-Mode")
