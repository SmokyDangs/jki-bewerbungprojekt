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
    st.image("https://www.julius-kuehn.de/fileadmin/templates/jki/img/jki_logo.png", width=150)
    st.markdown("### 🛠️ Konfiguration")
    
    conf_threshold = st.slider("KI-Konfidenz (Sensitivität)", 0.0, 1.0, 0.25, 0.05)
    
    st.divider()
    st.markdown("### 📊 Gesamtstatistik")
    sidebar_stats = st.container()

# --- HAUPTBEREICH ---
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("🔍 JKI Agroscan AI")
    st.markdown("*Intelligente Schaderreger-Erkennung für den modernen Pflanzenschutz*")

st.divider()

files = st.file_uploader(
    "Bilder hochladen (Drag & Drop)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if files:
    all_results_data = []
    
    # Fortschrittsanzeige
    progress_text = "Bilder werden prozessiert..."
    my_bar = st.progress(0, text=progress_text)

    # Überschrift für die Ergebnisse
    st.subheader(f"Analyse-Ergebnisse ({len(files)} Bilder)")

    # Raster-Layout: 2 Bilder pro Zeile
    img_cols = st.columns(2)
    
    for i, file in enumerate(files):
        # Fortschritt aktualisieren
        my_bar.progress((i + 1) / len(files), text=f"Analysiere {file.name} ({i+1}/{len(files)})")

        image = Image.open(file)
        # Inferenz durchführen
        results = model.predict(image, conf=conf_threshold)
        # Bounding Boxes einzeichnen (nutzt die deutschen Namen aus model.names)
        res_plotted = results[0].plot(line_width=2, font_size=1.5)
        
        # Daten sammeln für Statistik/CSV
        detections = results[0].boxes.cls.tolist()
        current_image_counts = {}
        
        for cls_id in detections:
            name = model.names[int(cls_id)]
            all_results_data.append({"Bild": file.name, "Fund": name})
            current_image_counts[name] = current_image_counts.get(name, 0) + 1

        # Bild im Raster anzeigen (abwechselnd Spalte 0 und 1)
        with img_cols[i % 2]:
            st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
            st.image(res_plotted, caption=f"Analyse: {file.name}", use_container_width=True)
            
            # Kurze Info unter dem Bild, was gefunden wurde
            if current_image_counts:
                found_items = [f"{count}x {label}" for label, count in current_image_counts.items()]
                st.caption(f"**Gefunden:** {', '.join(found_items)}")
            else:
                st.caption("Keine Objekte gefunden.")
            st.markdown('</div>', unsafe_allow_html=True)

    # Fortschrittsbalken entfernen
    my_bar.empty()

    # --- TABELLARISCHE AUSWERTUNG ---
    if all_results_data:
        st.divider()
        st.subheader("📋 Zusammenfassung der Befunde")
        df = pd.DataFrame(all_results_data)
        
        # Pivot für die Übersicht
        summary_df = df.groupby(['Fund']).size().reset_index(name='Anzahl Gesamt')
        
        col_table, col_download = st.columns([3, 1])
        with col_table:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with col_download:
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Ergebnisse als CSV speichern",
                data=csv,
                file_name='jki_analyse_ergebnisse.csv',
                mime='text/csv',
            )

        # Sidebar Update
        with sidebar_stats:
            for _, row in summary_df.iterrows():
                st.metric(label=row['Fund'], value=row['Anzahl Gesamt'])
            st.success(f"Total: {len(df)} Detektionen")

else:
    # Willkommens-Screen
    st.markdown("""
        <div style="background-color: #e6f0eb; padding: 30px; border-radius: 15px; border-left: 5px solid #005432;">
            <h3>Willkommen beim JKI Agroscan System</h3>
            <p>Laden Sie Fotos von Nutzpflanzen hoch, um eine automatische Diagnose von Schädlingen und Krankheiten zu erhalten.</p>
            <ul>
                <li>Unterstützt Batch-Upload (ganze Ordner möglich)</li>
                <li>Echtzeit-Visualisierung mit eingezeichneten Bounding Boxes</li>
                <li>Deutsche Bezeichnungen direkt im Bild</li>
                <li>Datenexport für Monitoring-Zwecke</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Corn_field_in_summer_Germany.jpg/1200px-Corn_field_in_summer_Germany.jpg", 
             caption="Überwachung landwirtschaftlicher Flächen", use_container_width=True)

# --- FOOTER ---
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption("© 2026 Julius Kühn-Institut (JKI) - Bundesforschungsinstitut für Kulturpflanzen")
with col_f2:
    st.markdown("<div style='text-align: right;'><small>System-Status: Betriebsbereit | KI-Kern: YOLOv8</small></div>", unsafe_allow_html=True)
