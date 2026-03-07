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

# --- BRANDING & CSS (Adaptive Design) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global Font & Spacing */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #005432 0%, #007d4a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Adaptive Card Design für Light & Dark Mode */
    .res-card {
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 20px;
        background-color: rgba(128, 128, 128, 0.05);
        transition: transform 0.2s;
    }
    .res-card:hover {
        transform: translateY(-2px);
        border-color: #007d4a;
    }
    
    /* Custom Badge für Befunde */
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
    # Placeholder für das echte Modell-Laden
    try:
        model = YOLO("best.pt")
        return model
    except:
        st.error("Modell 'best.pt' nicht gefunden. Bitte stellen Sie sicher, dass die Datei im Verzeichnis liegt.")
        return None

model = load_model()

def get_label(cls_id):
    if model and hasattr(model, 'names'):
        eng_name = model.names[int(cls_id)]
        return DEUTSCHE_NAMEN.get(eng_name, eng_name)
    return f"Klasse {cls_id}"

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.julius-kuehn.de/fileadmin/templates/jki/img/jki_logo.png", width=150) # Beispiel Logo-Platzhalter
    st.markdown("### 🔍 Agroscan AI")
    st.caption("v2.5 Enterprise Edition")
    st.divider()
    conf_threshold = st.slider("KI-Sensitivität (Confidence)", 0.0, 1.0, 0.25, 0.05)
    st.divider()
    sidebar_stats = st.container()

# --- HEADER ---
st.markdown(f"""
    <div class="main-header">
        <h1 style='margin:0; color:white;'>🌿 JKI Agroscan AI Dashboard</h1>
        <p style='margin:0; opacity:0.8;'>Batch-Diagnostik für den integrierten Pflanzenschutz</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Analyse & Monitoring", "📊 Analytics & Export"])

with tab1:
    files = st.file_uploader("Bilddaten hochladen", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files and model:
        all_results_data = []
        progress_bar = st.progress(0)
        
        # Grid-Layout für Ergebnisse
        cols = st.columns(3)
        
        for i, file in enumerate(files):
            # Bild laden
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Inferenz
            results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
            
            counts = {}
            if len(results[0].boxes) > 0:
                res_bgr = results[0].plot(line_width=2)
                res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
                
                detections = results[0].boxes.cls.tolist()
                confs = results[0].boxes.conf.tolist()
                
                for cls_id, conf in zip(detections, confs):
                    name = get_label(cls_id)
                    all_results_data.append({
                        "Bild": file.name, 
                        "Fund": name, 
                        "Konfidenz": round(conf, 2),
                        "Zeit": datetime.now().strftime("%H:%M:%S")
                    })
                    counts[name] = counts.get(name, 0) + 1
            else:
                res_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Darstellung in der Karte
            with cols[i % 3]:
                st.markdown('<div class="res-card">', unsafe_allow_html=True)
                st.image(res_rgb, use_container_width=True)
                st.caption(f"**Datei:** {file.name}")
                
                if counts:
                    # Verbesserte Darstellung der Befunde
                    html_badges = ""
                    for label, count in counts.items():
                        html_badges += f'<span class="badge">{count}x {label}</span>'
                    st.markdown(html_badges, unsafe_allow_html=True)
                else:
                    st.markdown("<small style='color:gray;'>Keine Schädlinge erkannt</small>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            progress_bar.progress((i + 1) / len(files))
        
        progress_bar.empty()
        st.session_state['analysis_data'] = all_results_data
    elif not model:
        st.warning("Das KI-Modell konnte nicht geladen werden.")
    else:
        st.info("Bitte laden Sie Bilddaten hoch, um die automatisierte Analyse zu starten.")

with tab2:
    if 'analysis_data' in st.session_state and st.session_state['analysis_data']:
        df = pd.DataFrame(st.session_state['analysis_data'])
        summary = df.groupby(['Fund']).size().reset_index(name='Anzahl')
        
        col_chart, col_data = st.columns([2, 1])
        
        with col_chart:
            # Theme-bewusste Charts
            fig = px.bar(
                summary, 
                x='Fund', 
                y='Anzahl', 
                color='Anzahl', 
                color_continuous_scale='Greens',
                template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
                title="Häufigkeit der Befunde"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col_data:
            st.subheader("Übersicht")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 CSV Report herunterladen",
                data=csv,
                file_name=f"JKI_Agroscan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            for _, row in summary.iterrows():
                st.metric(label=row['Fund'], value=int(row['Anzahl']))
        
        st.divider()
        st.subheader("Rohdaten")
        st.dataframe(df, use_container_width=True)
                
        with sidebar_stats:
            st.metric("Gesamtbefunde", len(df))
            st.metric("Einzigartige Arten", len(summary))
    else:
        st.warning("Noch keine Analysedaten verfügbar. Bitte führen Sie zuerst eine Analyse im ersten Tab durch.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<center><small style='opacity:0.6;'>© 2026 JKI | Pflanzenschutz-Diagnoseeinheit | Prototyp v2.5</small></center>", 
    unsafe_allow_html=True
)
