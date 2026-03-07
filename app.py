import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import os

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
    .stMetric { border: 1px solid #c8d6c8; padding: 15px; border-radius: 12px; background-color: #ffffff; }
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

# --- MODELL LADEN MIT FEHLER-CHECK ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    
    # Check ob Datei existiert
    if not os.path.exists(model_path):
        st.error(f"⚠️ Die Datei '{model_path}' wurde nicht gefunden! Bitte stellen Sie sicher, dass sie im Hauptverzeichnis liegt.")
        return None, None
    
    try:
        m = YOLO(model_path)
        # Deutsche Namen mappen
        translated_names = {id: DEUTSCHE_NAMEN.get(name.strip(), name.strip()) for id, name in m.names.items()}
        return m, translated_names
    except Exception as e:
        st.error(f"❌ Fehler beim Laden des Modells: {e}")
        st.info("Hinweis: Wenn Sie GitHub nutzen, könnte Git LFS das Problem sein. Die Datei 'best.pt' muss als echte Datei vorliegen, nicht als Text-Pointer.")
        return None, None

model, names_de = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.julius-kuehn.de/fileadmin/templates/jki/img/jki_logo.png", width=180)
    st.header("🔬 Parameter")
    conf_threshold = st.slider("KI-Sensitivität", 0.0, 1.0, 0.25, 0.05)
    
    st.divider()
    st.subheader("📊 Zusammenfassung")
    stats_placeholder = st.container()

# --- HAUPTBEREICH ---
st.title("🌱 JKI Crop & Pest Analyzer Pro")
st.markdown("Automatisierte Identifikation von Schaderregern | Julius Kühn-Institut")

if model is not None:
    uploaded_file = st.file_uploader("Pflanzenfoto hochladen...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        with st.spinner('Analysiere Bildmaterial...'):
            results = model.predict(image, conf=conf_threshold)
            
            # Temporärer Patch für deutsche Labels im Plot
            results[0].names = names_de 
            res_plotted = results[0].plot()
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Originalaufnahme", use_container_width=True)
            with col2:
                st.image(res_plotted, caption="Ergebnis mit deutschen Labels", use_container_width=True)

        # Auswertung
        boxes = results[0].boxes
        if len(boxes) > 0:
            df = pd.DataFrame([
                {"Objekt": names_de[int(box.cls[0])], "Sicherheit": float(box.conf[0])} 
                for box in boxes
            ])
            
            st.markdown('<div class="report-header">Analyse-Bericht</div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.dataframe(df.style.format({"Sicherheit": "{:.2%}"}), use_container_width=True, hide_index=True)
            with c2:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Bericht als CSV exportieren", csv, "jki_report.csv", "text/csv")
                st.metric("Durchschnittliche Sicherheit", f"{df['Sicherheit'].mean():.1%}")

            with stats_placeholder:
                for obj, count in df["Objekt"].value_counts().items():
                    st.metric(label=obj, value=int(count))
        else:
            st.warning("Keine Schädlinge im aktuellen Bild erkannt.")
else:
    st.warning("Das System ist aufgrund eines Modellfehlers derzeit nicht einsatzbereit.")

st.markdown("---")
st.caption("Forschungsprototyp | JKI Bewerbungsprojekt")
