import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="JKI Crop Detector", layout="wide")

st.title("🌱 JKI Prototyp: Crop & Weed Detector")
st.write("KI-gestützte Analyse von Nutzpflanzen und Beikräutern")

# Modell laden
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Pflanzenfoto hochladen...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Inferenz (Vorhersage)
    results = model(image)
    
    # Ergebnisse visualisieren
    res_plotted = results[0].plot()
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Originalbild", use_container_width=True)
    with col2:
        st.image(res_plotted, caption="KI-Erkennung", use_container_width=True)

    # Statistik ausgeben
    st.subheader("Analyse-Ergebnisse")
    detections = results[0].boxes.cls.tolist()
    names = model.names
    
    for idx, name in names.items():
        count = detections.count(idx)
        st.write(f"**{name.capitalize()}:** {count}")
