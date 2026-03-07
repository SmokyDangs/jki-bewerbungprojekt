import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import os
import datetime

# --- KONFIGURATION ---
st.set_page_config(
    page_title="JKI Crop & Pest Analyzer Pro",
    page_icon="🔬",
    layout="wide"
)

# Professionelles JKI-Branding CSS
st.markdown("""
    <style>
    .main { background-color: #f8faf8; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e1e4e8; }
    .stMetric { border: 1px solid #c8d6c8; padding: 15px; border-radius: 12px; background-color: #ffffff; box-shadow: 2px 2px 5px rgba(0,0,0,0.03); }
    .report-card { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #2e4d2e;
        margin-bottom: 20px;
    }
    .status-high { color: #d32f2f; font-weight: bold; }
    .status-low { color: #2e7d32; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Mapping-Dictionary (Deutsch)
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

@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        return None
    try:
        m = YOLO(model_path)
        return m
    except:
        return None

model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.julius-kuehn.de/fileadmin/templates/jki/img/jki_logo.png", width=180)
    st.header("⚙️ Analyse-Steuerung")
    conf_threshold = st.slider("KI-Konfidenz (Sensitivität)", 0.1, 1.0, 0.25, 0.05)
    
    st.divider()
    st.subheader("📊 Befund-Statistik")
    sidebar_stats = st.container()

# --- HAUPTBEREICH ---
st.title("🌱 JKI Crop & Pest Analyzer Pro")
st.write(f"Datum der Untersuchung: {datetime.date.today().strftime('%d.%m.%Y')}")

if model:
    uploaded_file = st.file_uploader("Bild zur Schaderreger-Analyse hochladen...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        with st.spinner('KI-Modell berechnet Befunde...'):
            # Inferenz
            results = model.predict(image, conf=conf_threshold)
            
            # --- DER DEUTSCH-FIX ---
            # Wir erzwingen die deutschen Namen im Plotting-Prozess
            # Wir erstellen ein Mapping-Dict basierend auf den IDs des Modells
            names_map = {id: DEUTSCHE_NAMEN.get(name, name) for id, name in model.names.items()}
            results[0].names = names_map # Temporäres Überschreiben für das Bild
            
            res_plotted = results[0].plot(line_width=2, font_size=1)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Originale Feldaufnahme", use_container_width=True)
            with col2:
                st.image(res_plotted, caption="Detektierte Schaderreger (DE)", use_container_width=True)

        # --- VERBESSERTER BERICHT ---
        st.markdown("---")
        st.subheader("📋 Digitaler Befundbericht")
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            # Daten sammeln
            data = []
            for box in boxes:
                cls_id = int(box.cls[0])
                name_de = names_map[cls_id]
                conf = float(box.conf[0])
                data.append({"Schaderreger": name_de, "Sicherheit": conf})
            
            df = pd.DataFrame(data)
            
            # Dashboard Layout
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.write("**Identifizierte Populationen:**")
                # Gruppierte Zusammenfassung
                summary_df = df.groupby("Schaderreger").agg(
                    Anzahl=("Schaderreger", "count"),
                    Ø_Sicherheit=("Sicherheit", "mean")
                ).reset_index()
                
                st.dataframe(summary_df.style.format({"Ø_Sicherheit": "{:.1%}"}), 
                             use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c2:
                # Befallsschwere einschätzen
                total_pests = len(df)
                risk_status = "HOCH" if total_pests > 10 else "MODERAT" if total_pests > 3 else "GERING"
                color_class = "status-high" if risk_status != "GERING" else "status-low"
                
                st.markdown(f"""
                <div class="report-card">
                    <p><strong>Befallsintensität:</strong></p>
                    <h2 class="{color_class}">{risk_status}</h2>
                    <p>Gesamtanzahl Funde: {total_pests}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # CSV Export
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Vollständigen Datensatz exportieren", csv, "jki_befund.csv", "text/csv")

            # Sidebar Update
            with sidebar_stats:
                for _, row in summary_df.iterrows():
                    st.metric(label=row["Schaderreger"], value=int(row["Anzahl"]))
        else:
            st.info("Keine Schaderreger im Analysebereich gefunden.")
else:
    st.error("Modell 'best.pt' fehlt. Bitte laden Sie die Modelldatei in das Hauptverzeichnis hoch.")

st.markdown("---")
st.caption("Forschungsprototyp | Julius Kühn-Institut (JKI) | Fachbereich Diagnostik")
