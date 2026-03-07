import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import io

# --- KONFIGURATION ---
st.set_page_config(
    page_title="JKI Crop & Pest Analyzer Pro",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS für JKI-Branding
st.markdown("""
    <style>
    .main { background-color: #f4f7f4; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e1e4e8; }
    .stMetric { border: 1px solid #c8d6c8; padding: 15px; border-radius: 12px; background-color: #ffffff; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .report-header { font-size: 24px; font-weight: bold; color: #2e4d2e; border-bottom: 2px solid #2e4d2e; margin-bottom: 20px; }
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

# --- MODELL LADEN (FIXED ATTRIBUTE ERROR) ---
@st.cache_resource
def load_model():
    m = YOLO("best.pt")
    # Wir speichern die übersetzten Namen in einer Variable, 
    # anstatt das geschützte Attribut direkt zu überschreiben
    original_names = m.names
    translated_names = {id: DEUTSCHE_NAMEN.get(name.strip(), name.strip()) for id, name in original_names.items()}
    return m, translated_names

# Modell und Übersetzungsliste laden
model, names_de = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.julius-kuehn.de/fileadmin/templates/jki/img/jki_logo.png", width=180)
    st.header("🔬 Analyse-Parameter")
    
    conf_threshold = st.slider("Sensitivität (Confidence)", 0.0, 1.0, 0.25, 0.05)
    show_labels = st.checkbox("Labels im Bild anzeigen", value=True)
    show_boxes = st.checkbox("Bounding Boxes anzeigen", value=True)
    
    st.divider()
    st.subheader("📊 Zusammenfassung")
    stats_container = st.container()

# --- HAUPTBEREICH ---
st.title("🌱 JKI Crop & Pest Analyzer Pro")
st.markdown("Automatisierte Identifikation von Schaderregern für den digitalen Pflanzenschutz.")

uploaded_file = st.file_uploader("Bild zur Analyse hochladen...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    with st.spinner('KI führt Präzisionsanalyse durch...'):
        results = model.predict(image, conf=conf_threshold)
        
        # Um die Labels im Bild auf Deutsch zu bekommen, 
        # überschreiben wir sie im Result-Objekt temporär für den Plot
        results[0].names = names_de 
        res_plotted = results[0].plot(labels=show_labels, boxes=show_boxes)
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(image, caption="Original-Input", use_container_width=True)
        with col_img2:
            st.image(res_plotted, caption="KI-Befund (Annotiert)", use_container_width=True)

    st.divider()

    # --- DATEN-AUSWERTUNG ---
    boxes = results[0].boxes
    if len(boxes) > 0:
        df_list = []
        for box in boxes:
            class_id = int(box.cls[0])
            name = names_de[class_id] # Nutzung der deutschen Liste
            confidence = float(box.conf[0])
            df_list.append({"Objekt": name, "Sicherheit": confidence})
        
        df = pd.DataFrame(df_list)
        
        st.markdown('<div class="report-header">Detaillierter Befundbericht</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(df.style.format({"Sicherheit": "{:.2%}"}), use_container_width=True, hide_index=True)
        
        with c2:
            st.write("### 📥 Daten-Export")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Befund als CSV speichern",
                data=csv,
                file_name="jki_analyse_bericht.csv",
                mime="text/csv"
            )
            avg_conf = df["Sicherheit"].mean()
            st.metric("Ø Analyse-Sicherheit", f"{avg_conf:.1%}")

        with stats_container:
            counts = df["Objekt"].value_counts()
            for obj_name, count in counts.items():
                st.metric(label=obj_name, value=int(count))
            st.info(f"Insgesamt {len(df)} Detektionen.")
    else:
        st.warning("Keine Schaderreger erkannt. Passen Sie die Sensitivität an.")

else:
    st.info("System bereit. Bitte Bildmaterial zur Analyse einspeisen.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Corn_field_in_summer_Germany.jpg/1200px-Corn_field_in_summer_Germany.jpg", 
             caption="Vorschau: Monitoring landwirtschaftlicher Nutzflächen", use_container_width=True)

st.markdown("---")
st.caption("Digitale Diagnostik v1.5 PRO | Julius Kühn-Institut (JKI) Bewerbungsprojekt")
