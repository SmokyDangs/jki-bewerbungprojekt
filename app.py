import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io

# --- KONFIGURATION ---
st.set_page_config(
    page_title="JKI Agroscan AI",
    page_icon="🔍",
    layout="wide"
)

# --- BRANDING & CSS ---
# JKI Farben: Dunkelgrün (~#005432), Hellgrün (~#86bc25)
st.markdown("""
    <style>
    /* Hintergrund und Schrift */
    .stApp { background-color: #f8fafc; }
    h1, h2, h3 { color: #005432; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    
    /* Karten-Design für Metriken */
    div[data-testid="stMetricValue"] { font-size: 24px; color: #005432; }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #005432;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover { background-color: #86bc25; border: none; color: white; }

    /* Stil für die Bild-Container */
    .image-container {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 10px;
        background-color: white;
        margin-bottom: 20px;
    }
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
    st.title("🌿 JKI Agroscan")
    st.subheader("🛠️ Analyse-Einstellungen")
    
    conf_threshold = st.slider("KI-Konfidenz (Sensitivität)", 0.0, 1.0, 0.25, 0.05,
                               help="Bestimmt, wie sicher sich die KI sein muss, um einen Fund anzuzeigen.")
    
    st.divider()
    st.markdown("### 📊 Live-Statistik")
    sidebar_stats = st.container()

# --- HAUPTBEREICH ---
st.header(":material/agriculture: Schaderreger-Diagnostik Prototyp")
st.markdown("*Bundesforschungsinstitut für Kulturpflanzen - Intelligentes Monitoring*")

st.divider()

files = st.file_uploader(
    "Bilder zur Analyse hochladen (Einzel- oder Batch-Modus)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    help="Ziehen Sie mehrere Bilder hierher, um eine Batch-Verarbeitung zu starten."
)

if files:
    all_results_data = []
    
    # Fortschrittsanzeige
    progress_text = "KI-Modell analysiert Bilddaten..."
    my_bar = st.progress(0, text=progress_text)

    st.subheader(f":material/view_module: Analyse-Ergebnisse ({len(files)} Bilder)")

    # Raster-Layout: 2 Bilder pro Zeile
    img_cols = st.columns(2)
    
    for i, file in enumerate(files):
        my_bar.progress((i + 1) / len(files), text=f"Verarbeite: {file.name}")

        image = Image.open(file)
        results = model.predict(image, conf=conf_threshold)
        res_plotted = results[0].plot(line_width=2, font_size=1.5)
        
        detections = results[0].boxes.cls.tolist()
        current_image_counts = {}
        
        for cls_id in detections:
            name = model.names[int(cls_id)]
            all_results_data.append({"Bild": file.name, "Fund": name})
            current_image_counts[name] = current_image_counts.get(name, 0) + 1

        with img_cols[i % 2]:
            st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
            st.image(res_plotted, caption=f"Identifikation: {file.name}", use_container_width=True)
            
            if current_image_counts:
                found_items = [f"**{count}x** {label}" for label, count in current_image_counts.items()]
                st.markdown(f"🔍 {' · '.join(found_items)}")
            else:
                st.caption("Keine signifikanten Funde auf diesem Bild.")
            st.markdown('</div>', unsafe_allow_html=True)

    my_bar.empty()

    # --- TABELLARISCHE AUSWERTUNG ---
    if all_results_data:
        st.divider()
        st.subheader(":material/analytics: Zusammenfassung der Befunde")
        df = pd.DataFrame(all_results_data)
        summary_df = df.groupby(['Fund']).size().reset_index(name='Anzahl Gesamt')
        
        col_table, col_download = st.columns([3, 1])
        with col_table:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with col_download:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Bericht herunterladen (CSV)",
                data=csv,
                file_name='jki_analyse_bericht.csv',
                mime='text/csv',
                use_container_width=True
            )

        with sidebar_stats:
            for _, row in summary_df.iterrows():
                st.metric(label=row['Fund'], value=int(row['Anzahl Gesamt']))
            st.success(f"Gesamt: {len(df)} Detektionen")

else:
    # Willkommens-Screen ohne Hero-Bild
    st.markdown("""
        <div style="background-color: #e6f0eb; padding: 40px; border-radius: 15px; border-left: 5px solid #005432; margin-top: 20px;">
            <h2 style="margin-top:0;">🚀 Startbereit</h2>
            <p>Bitte laden Sie ein oder mehrere Fotos hoch, um die KI-gestützte Analyse zu starten. Das System identifiziert automatisch:</p>
            <ul style="columns: 2;">
                <li>Schaderreger & Insekten</li>
                <li>Larvenstadien & Eier</li>
                <li>Pflanzenkrankheiten</li>
                <li>Nützlinge</li>
            </ul>
            <p><small>Tipp: Sie können auch einen ganzen Ordner mit Bildern markieren und hierher ziehen.</small></p>
        </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption("© 2026 Julius Kühn-Institut (JKI) | v1.4 Batch AI Core")
with col_f2:
    st.markdown("<div style='text-align: right;'><small>:material/check_circle: System-Status: Online | KI: YOLOv8</small></div>", unsafe_allow_html=True)
