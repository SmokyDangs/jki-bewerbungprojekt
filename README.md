🌿 JKI Agroscan AI | Enterprise Edition
KI-basierte Batch-Diagnostik für den integrierten Pflanzenschutz
JKI Agroscan AI ist ein hochspezialisiertes Analyse-Tool zur automatisierten Identifikation von Schädlingen und Krankheitserregern an Kulturpflanzen. Entwickelt als Prototyp (v2.5) für den Einsatz in der landwirtschaftlichen Diagnostik, kombiniert es modernste Computer Vision (YOLO) mit einer intuitiven Benutzeroberfläche für Batch-Analysen.

📋 Kernfunktionen
Automatisierte Batch-Analyse: Gleichzeitige Verarbeitung mehrerer Bilddateien zur effizienten Feld-Diagnostik.

Präzise Klassifizierung: Erkennung von über 25 spezifischen Schädlingen (Blattläuse, Kohlmotten, Wanzen, Larvenstadien etc.) mit deutscher Nomenklatur.

Interaktives Monitoring: Visualisierung der Befunde direkt im Bild mit dynamischen Confidence-Schwellenwerten.

Enterprise Analytics: Integriertes Dashboard mit Plotly-Charts zur statistischen Auswertung der Befallsdichte.

Export-Funktion: Generierung von CSV-Reports für die weiterführende wissenschaftliche Dokumentation.

🔬 Wissenschaftlicher Kontext
Das System unterstützt die Ziele des integrierten Pflanzenschutzes durch:

Früherkennung: Identifikation von Schädlingen bereits in frühen Entwicklungsstadien (Eier, Nymphen, Larven).

Monitoring: Präzise Erfassung der Befallsfrequenz zur Optimierung von Pflanzenschutzmaßnahmen.

Dokumentation: Revisionssichere Archivierung von Diagnose-Daten.

🛠️ Tech-Stack
Inferenz: Ultralytics YOLO (State-of-the-Art Object Detection).

Frontend: Streamlit (Adaptive UI für Light/Dark Mode).

Datenanalyse: Pandas & Plotly Express.

Bildverarbeitung: OpenCV & NumPy.

📦 Lokales Setup (Linux Mint / Debian)
Repository klonen:

Bash
git clone https://github.com/SmokyDangs/jki-bewerbungprojekt.git
cd jki-bewerbungprojekt
Abhängigkeiten installieren:
Hinweis: Nutzt die Headless-Version von OpenCV für Server-Kompatibilität.

Bash
pip install -r requirements.txt
Anwendung starten:

Bash
streamlit run app.py
📂 Struktur & Datenmapping
Die Anwendung nutzt ein Mapping-System, um internationale wissenschaftliche Bezeichnungen in gebräuchliche deutsche Namen zu übersetzen (z.B. Plagiodera versicolora → Weidenblattkäfer). Dies erhöht die Usability für Anwender im deutschsprachigen Raum erheblich.
