import streamlit as st
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
import plotly.express as px
from datetime import datetime
from PIL import Image

# --- KONFIGURATION ---
st.set_page_config(
    page_title="JKI Agroscan AI | Enterprise",
    page_icon="🌿",
    layout="wide"
)

# --- ERWEITERTES BRANDING & CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f1f5f9; }
    .main-header {
        background: linear-gradient(90deg, #005432 0%, #007d4a 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .res-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .stImage > img { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA & MAPPING ---
DEUTSCHE_NAMEN = {
    "Adulto": "Erwachsenes Tier", "Black-grass-caterpillar": "Schwarze Graseule (Raupe)",
    "Cerambycidae_larvae": "Bockkäfer-Larve", "Cnidocampa_flavescens-walker_pupa-": "Gelbe Schlupfwespe (Puppe)",
    "Coconut-black-headed-caterpillar": "Kokospalmen-Raupe", "Cricket": "Grille",
    "Diamondback-moth": "Kohlmotte", "Drosicha_contrahens_female": "Schildlaus (Weibchen)",
    "Erthesina_fullo_nymph-2": "Gelbgefleckte Baumwanze (Nymphe)", "Grasshopper": "Heuschrecke",
    "Hyphantria_cunea_larvae": "Amerikanischer Webebär (Larve)", "Hyphantria_cunea_pupa": "Amerikanischer Webebär (Puppe)",
    "Latoia_consocia_walker_larvae": "Schneckenspinner-Larve", "Leaf-eating-caterpillar": "Blattfressende Raupe",
    "Micromelalopha_troglodyta-graeser-_larvae": "Pappelspinner-Larve", "Ninfa": "Nymphe",
    "Ovo": "Ei", "Phenacoccus": "Schmierlaus", "Plagiodera_versicolora-laicharting-": "Weidenblattkäfer",
    "Plagiodera_versicolora-laicharting-_larvae": "Weidenblattkäfer (Larve)", "Plagiodera_versicolora-laicharting-_ovum": "Weidenblattkäfer (Ei)",
    "Red cotton steiner": "Rote Baumwollwanze", "Sericinus_montela_larvae": "Schwalbenschwanz-Raupe",
    "Spilarctia_subcarnea-walker-_larvae-2": "Bärenspinner-Raupe", "Wereng coklat": "Braune Zikade",
    "Aphid": "Blattlaus", "Citricola scale": "Zitrus-Schildlaus"
}

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
    st.markdown("## 🔍 Agroscan AI")
    st.caption("v2.2 Enterprise Edition")
    st.divider()
    conf_threshold = st.slider("KI-Konfidenz (Sensitivität)", 0.0, 1.0, 0.25, 0.05)
    # Neu: IoU Slider um Überlappungen besser zu handhaben
    iou_threshold = st.slider("Überlappungs-Toleranz (IoU)", 0.0, 1.0, 0.45, 0.05)
    st.divider()
    sidebar_stats = st.container()

# --- HEADER ---
st.markdown(f"""
    <div class="main-header">
        <h1>🌿 JKI Agroscan AI Dashboard</h1>
        <p>Präzisions-Monitoring: Batch-Verarbeitung mit optimierter Objekttrennung</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Analyse & Monitoring", "📊 Analytics & Export"])

with tab1:
    files = st.file_uploader("Bilddaten hochladen", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files:
        all_results_data = []
        progress_bar = st.progress(0)
        cols = st.columns(3)
        
        for i, file in enumerate(files):
            # Bild laden
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # KI Inferenz mit Fix für Mehrfach-Markierungen:
            # iou=iou_threshold erlaubt eng beieinander liegende Boxen
            # agnostic_nms=True verhindert, dass verschiedene Klassen sich gegenseitig löschen
            results = model.predict(
                img_bgr, 
                conf=conf_threshold, 
                iou=iou_threshold, 
                agnostic_nms=True
            )
            
            # Plotten mit etwas dünneren Linien für kleine Objekte (Blattläuse)
            res_bgr = results[0].plot(line_width=1, font_size=0.8, labels=True)
            res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
            
            # Statistik
            if results[0].boxes:
                detections = results[0].boxes.cls.tolist()
                counts = {}
                for cls_id in detections:
                    name = model.names[int(cls_id)]
                    all_results_data.append({"Bild": file.name, "Fund": name, "Zeit": datetime.now().strftime("%H:%M:%S")})
                    counts[name] = counts.get(name, 0) + 1

                with cols[i % 3]:
                    st.markdown('<div class="res-card">', unsafe_allow_html=True)
                    st.image(res_rgb, use_container_width=True)
                    st.caption(f"📄 {file.name}")
                    for label, count in counts.items():
                        st.markdown(f"**{count}x** :green[{label}]")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                with cols[i % 3]:
                    st.markdown('<div class="res-card">', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption(f"📄 {file.name}")
                    st.markdown(":gray[Keine Schädlinge gefunden]")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            progress_bar.progress((i + 1) / len(files))
        
        progress_bar.empty()
        st.session_state['analysis_data'] = all_results_data
    else:
        st.info("Bereit für Upload. Bitte Bilder auswählen.")

with tab2:
    if 'analysis_data' in st.session_state and st.session_state['analysis_data']:
        df = pd.DataFrame(st.session_state['analysis_data'])
        summary = df.groupby(['Fund']).size().reset_index(name='Anzahl')
        
        col_chart, col_data = st.columns([2, 1])
        with col_chart:
            fig = px.bar(summary, x='Fund', y='Anzahl', color='Anzahl', color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        with col_data:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Report herunterladen", csv, "jki_report.csv", "text/csv", use_container_width=True)
            for _, row in summary.iterrows():
                st.metric(label=row['Fund'], value=int(row['Anzahl']))
        with sidebar_stats:
            st.metric("Gesamtbefunde", len(df))
    else:
        st.warning("Noch keine Daten vorhanden.")

st.markdown("---")
st.markdown("<center><small>© 2026 JKI | Prototyp v2.2 | Fix: Multi-Object-Detection</small></center>", unsafe_allow_html=True)
