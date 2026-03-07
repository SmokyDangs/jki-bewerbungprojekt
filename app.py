import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import cv2

# --- KONFIGURATION ---
st.set_page_config(
    page_title="JKI Agroscan AI",
    page_icon="🔍",
    layout="wide"
)

# --- BRANDING & CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    h1, h2, h3 { color: #005432; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    div[data-testid="stMetricValue"] { font-size: 22px; color: #005432; font-weight: bold; }
    
    .image-container {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 10px;
        background-color: white;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stImage > img { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA & MAPPING ---
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
    model = YOLO("best.pt")
    if hasattr(model, 'names'):
        for idx, eng_name in model.names.items():
            if eng_name in DEUTSCHE_NAMEN:
                model.names[idx] = DEUTSCHE_NAMEN[eng_name]
    return model

model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🌿 JKI Agroscan")
    st.divider()
    conf_threshold = st.slider("KI-Konfidenz (Sensitivität)", 0.0, 1.0, 0.25, 0.05)
    st.divider()
    sidebar_stats = st.container()

# --- HAUPTBEREICH ---
st.header(":material/agriculture: Schaderreger-Diagnostik")
st.markdown("*KI-gestütztes Monitoring für den Pflanzenschutz*")

files = st.file_uploader("Bilder hochladen", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if files:
    all_results_data = []
    my_bar = st.progress(0)
    img_cols = st.columns(2)
    
    for i, file in enumerate(files):
        my_bar.progress((i + 1) / len(files), text=f"Analysiere: {file.name}")

        # 1. Bild laden und direkt in das von YOLO bevorzugte Format bringen
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 2. Inferenz (YOLO bekommt direkt das BGR-Array aus OpenCV)
        results = model.predict(img_bgr, conf=conf_threshold)
        
        # 3. Plotten (YOLO plottet auf das BGR-Bild)
        # img_size sorgt dafür, dass die Boxen im GUI scharf gezeichnet werden
        res_bgr = results[0].plot(line_width=2, font_size=1.2, labels=True)
        
        # 4. Zurück nach RGB für die Streamlit-Anzeige
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        
        detections = results[0].boxes.cls.tolist()
        current_image_counts = {}
        
        for cls_id in detections:
            name = model.names[int(cls_id)]
            all_results_data.append({"Bild": file.name, "Fund": name})
            current_image_counts[name] = current_image_counts.get(name, 0) + 1

        with img_cols[i % 2]:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(res_rgb, caption=f"Ergebnis: {file.name}", use_container_width=True)
            
            if current_image_counts:
                labels = [f"**{count}x** {label}" for label, count in current_image_counts.items()]
                st.markdown(f"✅ {' · '.join(labels)}")
            else:
                st.caption("Keine Schädlinge erkannt.")
            st.markdown('</div>', unsafe_allow_html=True)

    my_bar.empty()

    if all_results_data:
        st.divider()
        df = pd.DataFrame(all_results_data)
        summary = df.groupby(['Fund']).size().reset_index(name='Anzahl')
        c1, c2 = st.columns([3, 1])
        c1.dataframe(summary, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode('utf-8')
        c2.download_button("📥 CSV Export", csv, "jki_report.csv", "text/csv")

        with sidebar_stats:
            for _, row in summary.iterrows():
                st.metric(label=row['Fund'], value=int(row['Anzahl']))
            st.success(f"Gesamt: {len(df)} Funde")
else:
    st.info("Bitte laden Sie Bilder hoch, um die Analyse zu starten.")

st.markdown("---")
st.markdown("<center><small>© 2026 JKI | Prototyp v1.7 | Optimierte Modell-Übergabe</small></center>", unsafe_allow_html=True)
