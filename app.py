import streamlit as st
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
import plotly.express as px
from datetime import datetime

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
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #005432 0%, #007d4a 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Card Styling */
    .res-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
    }
    .res-card:hover { transform: translateY(-5px); box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
    
    /* Sidebar adjustments */
    [data-testid="stSidebar"] { background-color: #ffffff; }
    .stSlider [data-baseweb="slider"] { margin-bottom: 2rem; }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] { color: #005432; font-weight: 700; }
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
    st.caption("v2.0 Enterprise Edition")
    st.divider()
    
    with st.expander("⚙️ Parameter", expanded=True):
        conf_threshold = st.slider("KI-Sensitivität", 0.0, 1.0, 0.25, 0.05)
        st.info("Höhere Sensitivität erkennt mehr, kann aber Fehlalarme auslösen.")
    
    st.divider()
    st.markdown("### 📈 Live-Counter")
    sidebar_stats = st.container()

# --- HAUPTBEREICH HEADER ---
st.markdown(f"""
    <div class="main-header">
        <h1>🌿 JKI Agroscan AI Dashboard</h1>
        <p>Automatisierte Identifikation von Schaderregern im integrierten Pflanzenschutz</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Analyse & Monitoring", "📊 Analytics & Export"])

with tab1:
    files = st.file_uploader("Bilddaten für Batch-Analyse hochladen", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files:
        all_results_data = []
        progress_bar = st.progress(0)
        
        # Grid Layout
        cols = st.columns(3) # Auf 3 Spalten erhöht für bessere Übersicht bei vielen Bildern
        
        for i, file in enumerate(files):
            # Inferenz Prozess
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            results = model.predict(img_bgr, conf=conf_threshold)
            
            # Plotting & Conversion
            res_bgr = results[0].plot(line_width=2, font_size=1.0, labels=True)
            res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
            
            # Data Collection
            detections = results[0].boxes.cls.tolist()
            counts = {}
            for cls_id in detections:
                name = model.names[int(cls_id)]
                all_results_data.append({"Bild": file.name, "Fund": name, "Zeit": datetime.now().strftime("%H:%M:%S")})
                counts[name] = counts.get(name, 0) + 1

            # UI Rendering in Cards
            with cols[i % 3]:
                st.markdown('<div class="res-card">', unsafe_allow_html=True)
                st.image(res_rgb, use_container_width=True)
                st.caption(f"📄 {file.name}")
                
                if counts:
                    for label, count in counts.items():
                        st.markdown(f"**{count}x** :green[{label}]")
                else:
                    st.markdown(":gray[Keine Befunde]")
                st.markdown('</div>', unsafe_allow_html=True)
            
            progress_bar.progress((i + 1) / len(files))
        
        progress_bar.empty()
        st.toast(f"Analyse von {len(files)} Bildern abgeschlossen!", icon='✅')

    else:
        st.info("Bereit für Upload. Ziehen Sie Bilder hierher, um das Monitoring zu starten.")

with tab2:
    if 'all_results_data' in locals() and all_results_data:
        df = pd.DataFrame(all_results_data)
        summary = df.groupby(['Fund']).size().reset_index(name='Anzahl')
        
        col_l, col_r = st.columns([2, 1])
        
        with col_l:
            st.subheader("Befundverteilung")
            fig = px.bar(summary, x='Fund', y='Anzahl', color='Anzahl', 
                         color_continuous_scale='Greens', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Rohdaten")
            st.dataframe(df, use_container_width=True, hide_index=True)
            
        with col_r:
            st.subheader("Export")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Full Report (CSV)",
                data=csv,
                file_name=f'JKI_Report_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
                use_container_width=True
            )
            
            st.divider()
            st.markdown("### Schnell-Statistik")
            for _, row in summary.iterrows():
                st.metric(label=row['Fund'], value=int(row['Anzahl']))

        # Sidebar Stats Update
        with sidebar_stats:
            st.metric("Gesamtbefunde", len(df))
            st.success(f"Bilder prozessiert: {len(files)}")
    else:
        st.warning("Keine Daten vorhanden. Bitte führen Sie zuerst eine Analyse im Tab 'Analyse' durch.")

# --- FOOTER ---
st.markdown("---")
f_left, f_right = st.columns(2)
with f_left:
    st.caption("© 2026 Julius Kühn-Institut | Forschungs-Prototyp AI-Detection")
with f_right:
    st.markdown("<div style='text-align: right;'><small>Status: System Online 🟢</small></div>", unsafe_allow_html=True)
