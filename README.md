# <p align="center">🌿 JKI Agroscan AI | Enterprise Edition</p>

<p align="center">
  <a href="https://jki-agroscan.streamlit.app/">
    <img src="https://img.shields.io/badge/LIVE_DEMO-JETZT_TESTEN-007d4a?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-2.5_Enterprise-007d4a?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/AI-YOLOv8/v10-00FFFF?style=flat-square" alt="YOLO">
  <img src="https://img.shields.io/badge/Data-Roboflow-6706ce?style=flat-square&logo=roboflow" alt="Roboflow">
</p>

---

## 📖 Projekt-Übersicht
**JKI Agroscan AI** ist ein hochspezialisiertes Analyse-Tool zur automatisierten Identifikation von Schädlingen und Krankheitserregern an Kulturpflanzen. Entwickelt als **Prototyp (v2.5)** für den Einsatz in der landwirtschaftlichen Diagnostik am **Julius Kühn-Institut**, kombiniert es modernste **Computer Vision (YOLO)** mit einer intuitiven Batch-Processing-Oberfläche.

> [!IMPORTANT]
> **Live-Anwendung:** Die App ist unter [jki-agroscan.streamlit.app](https://jki-agroscan.streamlit.app/) öffentlich erreichbar und für den **integrierten Pflanzenschutz** optimiert.

---

## 📋 Kernfunktionen

| Feature | Beschreibung |
| :--- | :--- |
| **🚀 Batch-Analyse** | Gleichzeitige Verarbeitung hunderter Bilddaten zur effizienten Feld-Diagnostik. |
| **🔬 Präzise Klassifizierung** | Erkennung von 12 Arten (Blattläuse, Kohlmotten, Wanzen) inkl. Larvenstadien. |
| **🔍 Live-Monitoring** | Sofortige Visualisierung mit dynamischen Confidence-Filtern (KI-Sensitivität). |
| **📊 Enterprise Analytics** | Interaktive Plotly-Dashboards zur statistischen Befallsauswertung und Frequenzanalyse. |
| **📥 Export-System** | Generierung wissenschaftlicher CSV-Reports für die revisionssichere Dokumentation. |

---

## 🔬 Wissenschaftlicher Kontext & Modell-Training

Das System transformiert die klassische Schädlingsbestimmung in einen digitalen Hochgeschwindigkeits-Workflow. Das Fine-Tuning des YOLO-Modells basiert auf qualitativ hochwertigen agrarwissenschaftlichen Daten.

### 📊 Datensatz & Training
Für das Training und die Validierung der KI wurde der folgende spezialisierte Datensatz verwendet:
* **Dataset:** [Pest-Uruhn auf Roboflow Universe](https://universe.roboflow.com/dense-pset/pest-uruhn)
* **Inhalt:** Annotierte Bilddaten von Schädlingen in verschiedenen Entwicklungsstadien (Eier, Nymphen, Larven, Adult).



### Impact:
* **Früherkennung:** Identifikation kleinster Merkmale zur Prävention massiver Ernteausfälle.
* **Monitoring:** Objektive Erfassung der Befallsfrequenz zur präzisen Steuerung von Pflanzenschutzmaßnahmen.
* **Dokumentation:** Automatisierte Erstellung digitaler Befund-Protokolle für die Agrarforschung.

---

## 🛠️ Tech-Stack & Architektur

* **Core AI:** `ultralytics` YOLO (Segmentierung & Objekt-Erkennung).
* **UI/UX:** `streamlit` mit Enterprise-Custom-CSS (Inter-Font & Adaptive Design).
* **Data Science:** `pandas`, `plotly-express`, `numpy`.
* **Vision-Backend:** `opencv-python-headless` (Server-optimiert).

---

## 📦 Lokales Setup (Linux Mint / Debian)

### 1. Repository klonen
```bash
git clone [https://github.com/SmokyDangs/jki-bewerbungprojekt.git](https://github.com/SmokyDangs/jki-bewerbungprojekt.git)
cd jki-bewerbungprojekt
