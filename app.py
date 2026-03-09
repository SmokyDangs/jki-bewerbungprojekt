from ultralytics import solutions

# Initialisierung der fertigen Ultralytics-Streamlit-Lösung
# model: Pfad zu deiner 'best.pt'
inf = solutions.Inference(model="best.pt")

# Startet die Weboberfläche
inf.inference()

# WICHTIG: Starte diese Datei über das Terminal mit:
# streamlit run app.py
