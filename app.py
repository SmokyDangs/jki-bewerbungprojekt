import streamlit as st
from ultralytics import YOLO
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from PIL import Image
import io

# --- KONFIGURATION ---
st.set_page_config(
    page_title="JKI Agroscan AI | Enterprise",
    page_icon="🌿",
    layout="wide"
)

# --- BRANDING & CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(90deg, #005432 0%, #007d4a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .res-card {
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 20px;
        background-color: rgba(128, 128, 128, 0.05);
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        background: #007d4a;
        color: white;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
    }
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

# --- MODELL-LADEN ---
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Modell-Fehler: {e}")
        return None

model = load_model()

def get_label(cls_id):
    if model and hasattr(model, 'names'):
        eng_name = model.names[int(cls_id)]
        return DEUTSCHE_NAMEN.get(eng_name, eng_name)
    return f"Klasse {cls_id}"

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.julius-kuehn.de/fileadmin/templates/jki/img/jki_logo.png", width=150)
    st.markdown("### 🔍 Agroscan AI")
    conf_threshold = st.slider("KI-Sensitivität", 0.0, 1.0, 0.25, 0.05)
    sidebar_stats = st.container()

# --- HEADER ---
st.markdown('<div class="main-header"><h1 style="color:white; margin:0;">🌿 JKI Agroscan AI</h1></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Analyse", "📊 Statistik"])

# --- TAB 1: ANALYSE ---
with tab1:
    files = st.file_uploader("Bilder hochladen", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files and model:
        all_results_data = []
        cols = st.columns(3)
        
        for i, file in enumerate(files):
            # Ersetzung von cv2 durch PIL
            img = Image.open(file).convert("RGB")
            
            # KI-Inferenz (YOLO akzeptiert PIL Images direkt)
            results = model.predict(img, conf=conf_threshold, verbose=False)
            
            counts = {}
            if len(results[0].boxes) > 0:
                # results[0].plot() gibt ein numpy array (BGR) zurück
                # Wir konvertieren es direkt in ein PIL Image zur Anzeige
                res_array = results[0].plot() 
                # Da YOLO .plot() intern immer noch BGR nutzt, müssen wir die Kanäle tauschen:
                res_img = Image.fromarray(res_array[:, :, ::-1]) 
                
                for box in results[0].boxes:
                    name = get_label(box.cls[0])
                    all_results_data.append({
                        "Bild": file.name, "Fund": name, 
                        "Konfidenz": float(box.conf[0]),
                        "Zeit": datetime.now().strftime("%H:%M:%S")
                    })
                    counts[name] = counts.get(name, 0) + 1
            else:
                res_img = img

            with cols[i % 3]:
                st.markdown('<div class="res-card">', unsafe_allow_html=True)
                st.image(res_img, use_container_width=True)
                st.caption(f"Datei: {file.name}")
                if counts:
                    badges = "".join([f'<span class="badge">{c}x {l}</span>' for l, c in counts.items()])
                    st.markdown(badges, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.session_state['analysis_data'] = all_results_data

# --- TAB 2: STATISTIK ---
with tab2:
    if 'analysis_data' in st.session_state and st.session_state['analysis_data']:
        df = pd.DataFrame(st.session_state['analysis_data'])
        summary = df.groupby('Fund').size().reset_index(name='Anzahl')
        
        fig = px.bar(summary, x='Fund', y='Anzahl', color='Anzahl', color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
        
        with sidebar_stats:
            st.metric("Gesamtbefunde", len(df))
    else:
        st.info("Keine Daten vorhanden.")
