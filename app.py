import cv2
from ultralytics.solutions.streamlit_inference import Inference

def main():
    # Übergabe deines spezifischen Modells 'best.pt'
    # Die Inference-Klasse lädt dieses Modell automatisch in das Streamlit-UI
    inf = Inference(model="best.pt")

    # Startet die Weboberfläche und die Logik für Bilder, Videos oder Webcam
    inf.inference()

if __name__ == "__main__":
    main()
