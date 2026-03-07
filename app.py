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
    .stMetric { background-color: #ffffff; border: 1px solid #e1e4e8; padding: 15px; border-radius: 10px; }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
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

# Sidebar
with st.sidebar:
    st.header("Analyse-Optionen")
    conf_threshold = st.slider("KI-Konfidenz (Schwellenwert)", 0.0, 1.0, 0.25, 0.05, 
                               help="Bestimmt, ab welcher Wahrscheinlichkeit ein Objekt als erkannt gilt.")
    st.divider()
    st.markdown("### Über dieses Projekt")
    st.write("Entwickelt für das Monitoring invasiver Schädlinge und die automatisierte Felderfassung.")

# Hauptbereich
st.title("🌱 JKI Prototyp: Crop & Weed Detector")
st.markdown("#### Computer-Vision-gestützte Diagnostik für den modernen Pflanzenschutz")
st.divider()

uploaded_file = st.file_uploader("Bild zur Analyse hochladen (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner('Bildanalyse läuft...'):
        image = Image.open(uploaded_file)
        
        # Inferenz
        results = model.predict(image, conf=conf_threshold)
        res_plotted = results[0].plot()
        
        # Anzeige
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Originalaufnahme", use_container_width=True)
        with col2:
            st.image(res_plotted, caption="Visualisierte Erkennung", use_container_width=True)

        st.divider()
        
        # Ergebnisse
        st.subheader("Befundbericht")
        
        detections = results[0].boxes.cls.tolist()
        names = model.names
        
        found_data = []
        metric_cols = st.columns(4)
        m_idx = 0
        
        for idx, name in names.items():
            count = detections.count(idx)
            if count > 0:
                name_de = DEUTSCHE_NAMEN.get(name, name)
                with metric_cols[m_idx % 4]:
                    st.metric(label=name_de, value=int(count))
                m_idx += 1
                found_data.append({"Klasse": name_de, "Anzahl": int(count)})

        if found_data:
            with st.expander("Tabellarische Detailansicht"):
                df = pd.DataFrame(found_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"Keine Objekte über dem Schwellenwert von {conf_threshold} erkannt.")

else:
    # Willkommensansicht ohne den fehlerhaften alpha-Parameter
    st.info("Bitte laden Sie ein Foto hoch, um die Analyse zu starten.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Corn_field_in_summer_Germany.jpg/1200px-Corn_field_in_summer_Germany.jpg", 
             caption="Beispielhafte Feldaufnahme für das KI-Monitoring", 
             use_container_width=True)

# Footer
st.markdown("---")
st.caption("Forschungsprototyp | Julius Kühn-Institut (JKI) Bewerbungsprojekt")
