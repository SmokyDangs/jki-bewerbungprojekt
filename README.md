# <p align="center">🌿 JKI Agroscan AI | Enterprise Edition</p>
<p align="center">
  <img src="https://img.shields.io/badge/Version-2.5_Enterprise-007d4a?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/AI-YOLOv8/v10-00FFFF?style=for-the-badge" alt="YOLO">
</p>

---

## 📖 Projekt-Übersicht
**JKI Agroscan AI** ist ein hochspezialisiertes Analyse-Tool zur automatisierten Identifikation von Schädlingen und Krankheitserregern an Kulturpflanzen. Entwickelt als **Prototyp (v2.5)** für den Einsatz in der landwirtschaftlichen Diagnostik am **Julius Kühn-Institut**, kombiniert es modernste **Computer Vision (YOLO)** mit einer intuitiven Batch-Processing-Oberfläche.

> [!IMPORTANT]
> Diese App ist für den **integrierten Pflanzenschutz** optimiert und unterstützt Wissenschaftler dabei, Befallsdichten in Echtzeit zu quantifizieren.

---

## 📋 Kernfunktionen

| Feature | Beschreibung |
| :--- | :--- |
| **🚀 Batch-Analyse** | Gleichzeitige Verarbeitung hunderter Bilddaten zur Feld-Diagnostik. |
| **🔬 Präzise Klassifizierung** | Erkennung von >25 Arten (Blattläuse, Kohlmotten, Wanzen) inkl. Larvenstadien. |
| **🔍 Live-Monitoring** | Sofortige Visualisierung mit dynamischen Confidence-Filtern. |
| **📊 Enterprise Analytics** | Interaktive Plotly-Dashboards zur statistischen Befallsauswertung. |
| **📥 Export-System** | Generierung wissenschaftlicher CSV-Reports für die Dokumentation. |

---

## 🔬 Wissenschaftlicher Kontext & Impact

Das System transformiert die klassische Schädlingsbestimmung in einen digitalen Workflow:

* **Früherkennung:** Identifikation kleinster Merkmale (Eier, Nymphen) zur Prävention.
* **Monitoring:** Objektive Erfassung der Frequenz zur Optimierung von Pflanzenschutzmitteln.
* **Dokumentation:** Revisionssichere Archivierung digitaler Befunde.



---

## 🛠️ Tech-Stack & Architektur

* **Core AI:** `ultralytics` YOLO (Inferenz-Optimiert).
* **UI/UX:** `streamlit` mit Adaptive Design (Light/Dark Mode Support).
* **Data Science:** `pandas`, `plotly-express`.
* **Vision:** `opencv-python-headless` & `NumPy`.

---

## 📦 Lokales Setup (Linux Mint / Debian)

### 1. Repository klonen
```bash
git clone [https://github.com/SmokyDangs/jki-bewerbungprojekt.git](https://github.com/SmokyDangs/jki-bewerbungprojekt.git)
cd jki-bewerbungprojekt
