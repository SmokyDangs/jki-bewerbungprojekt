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

# JKI-Branding CSS
st.markdown("""
    <style>
    .main { background-color: #f8faf8; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e1e4e8; }
    .stMetric { border: 1px solid #c8d6c8; padding: 15px; border-radius: 12px; background-color: #ffffff; }
    .report-card { 
        background-color: #ffffff; padding: 20px; border-radius: 10px; 
        border-left: 5px solid #2e4d2e; margin-bottom: 20px;
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
    "Cricket": "Grille",
    "Diamondback-moth": "Kohlmotte",
    "Grasshopper": "Heuschrecke",
    "Ninfa": "Nymphe",
    "Ovo": "Ei",
    "Aphid": "Blattlaus",
    "Citricola scale": "Zitrus-Schildlaus"
}

# --- MODELL LADEN ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Datei '{model_path}' nicht gefunden!")
        st.info(f"Sichtbare Dateien: {os.listdir('.')}")
        return None
    try:
        # Modell laden, OHNE die Namen hier zu ändern (verursacht den Fehler)
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Kritischer Fehler beim Laden: {e}")
        return None

model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.julius-kuehn.de/fileadmin/templates/jki/img/jki_logo.png", width=180)
    st.header("⚙️ Analyse-Steuerung")
    conf_threshold = st.slider("KI-Konfidenz", 0.1, 1.0, 0.25, 0.05)
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
        
        with st.spinner('KI analysiert Bildmaterial...'):
            results = model.predict(image, conf=conf_threshold)
            
            # --- DER FIX FÜR DEN ATTRIBUTE-ERROR ---
            # Wir mappen die Namen NUR im aktuellen Ergebnis-Objekt für die Anzeige
            res = results[0]
            # Wir erstellen ein lokales Dictionary mit deutschen Namen für dieses spezifische Bild
            local_names = {id: DEUTSCHE_NAMEN.get(name, name) for id, name in res.names.items()}
            
            # Wir nutzen die .plot() Funktion und übergeben die Namen explizit, 
            # falls möglich, oder patchen das Objekt lokal
            res.names = local_names # Im Result-Objekt ist es meist erlaubt, im Modell-Objekt nicht!
            
            res_plotted = res.plot(line_width=2)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Originalaufnahme", use_container_width=True)
            with col2:
                st.image(res_plotted, caption="Befund (Deutsche Labels)", use_container_width=True)

        # --- VERBESSERTER BERICHT ---
        st.markdown("---")
        st.subheader("📋 Digitaler Befundbericht")
        
        boxes = res.boxes
        if len(boxes) > 0:
            data = []
            for box in boxes:
                cls_id = int(box.cls[0])
                # Nutze unser lokales Mapping
                label = local_names[cls_id]
                conf = float(box.conf[0])
                data.append({"Schaderreger": label, "Sicherheit": conf})
            
            df = pd.DataFrame(data)
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                summary_df = df.groupby("Schaderreger").agg(
                    Anzahl=("Schaderreger", "count"), 
                    Ø_Sicherheit=("Sicherheit", "mean")
                ).reset_index()
                st.dataframe(summary_df.style.format({"Ø_Sicherheit": "{:.1%}"}), 
                             use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c2:
                total = len(df)
                status = "HOCH" if total > 10 else "MODERAT" if total > 3 else "GERING"
                color = "status-high" if status != "GERING" else "status-low"
                st.markdown(f"""
                <div class="report-card">
                    <p>Befallsintensität:</p>
                    <h2 class="{color}">{status}</h2>
                    <p>Detektionen gesamt: {total}</p>
                </div>
                """, unsafe_allow_html=True)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 CSV Export", csv, "jki_befund.csv", "text/csv")

            with sidebar_stats:
                for _, row in summary_df.iterrows():
                    st.metric(label=row["Schaderreger"], value=int(row["Anzahl"]))
        else:
            st.info("Keine Schaderreger im gewählten Konfidenzbereich gefunden.")
else:
    st.warning("Das System konnte nicht initialisiert werden. Bitte prüfen Sie die Log-Dateien.")

st.markdown("---")
st.caption("Forschungsprototyp | Julius Kühn-Institut (JKI)")
