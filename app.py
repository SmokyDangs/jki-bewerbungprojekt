import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image
import numpy as np

st.set_page_config(page_title="JKI Zero-Shot Detector", layout="wide")

st.title("🔬 JKI Research: Zero-Shot Plant Discovery")
st.write("Identifikation von Objekten per Texteingabe ohne Training.")

# 1. Modell laden (v2 ist stabiler für Deployment)
@st.cache_resource
def load_yolo_world():
    return YOLOWorld("yolov8s-worldv2.pt")

model = load_yolo_world()

# 2. Sidebar für die dynamischen Klassen
st.sidebar.header("Analyse-Parameter")
user_prompt = st.sidebar.text_input("Suchbegriffe (kommagetrennt):", "aphid, leaf spot, sugar beet")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# Begriffe setzen (Die Magie von YOLO-World)
custom_classes = [c.strip() for c in user_prompt.split(",")]
model.set_classes(custom_classes)

# 3. Bild-Upload
uploaded_file = st.file_uploader("Bild zur Analyse hochladen...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    
    # Inferenz ausführen
    results = model.predict(img, conf=conf_threshold)
    
    # Visualisierung
    res_plotted = results[0].plot()
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Originalaufnahme", use_container_width=True)
    with col2:
        st.image(res_plotted, caption=f"Erkennung: {user_prompt}", use_container_width=True)

    # 4. Ergebnisauswertung
    st.subheader("Gefundene Objekte")
    counts = {}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = custom_classes[cls_id]
        counts[label] = counts.get(label, 0) + 1
    
    if counts:
        for label, count in counts.items():
            st.metric(label, count)
    else:
        st.info("Keine Objekte mit der gewählten Confidence gefunden.")
