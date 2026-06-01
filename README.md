#  Video Analysis Pipeline

Pipeline multimodal complet pour l'analyse automatique de vidéos : extraction de frames, OCR (RU/EN), transcription audio (Whisper), détection d'objets (YOLOv8 + Faster R-CNN), analyse sémantique (LLM Qwen-VL) et post-traitement intelligent.

## 📊 Architecture du Pipeline

mermaid
graph TD
    A[ Vidéo d'entrée] --> B(PZ2: Extraction des frames 1 FPS)
    B --> C[PZ3: EasyOCR RU/EN]
    B --> D[PZ4: Whisper Audio]
    B --> E[PZ5: YOLOv8n]
    B --> F[PZ6: Faster R-CNN ResNet50]
    B --> G[PZ7: Qwen-VL LLM]
    C & D & E & F & G --> H(PZ8: Post-traitement & Fusion)
    H --> I[📊 report_final.json]
    H --> J[📈 report_final.xlsx]


## 🚀 Installation & Exécution

ash
git clone https://github.com/Exenia-vavy/video-analysis-pipeline.git
cd video-analysis-pipeline
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py


##  Interface Web (Streamlit)
ash
python -m streamlit run app.py

- Upload vidéo → Lancement automatique
- Logs temps réel + téléchargement JSON/Excel
- Fonctionne 100% en local (Windows/Linux)

##  Structure
	ext
├── pz1_ocr.py → pz8_report.py  # 8 modules autonomes
├── run_pipeline.py              # Orchestrateur séquentiel
── app.py                       # Interface Streamlit
── requirements.txt             # Dépendances Python
├── Dockerfile                   # Prêt pour déploiement Linux
└── .env.example                 # Template configuration


##  Résultats
- **170 événements** consolidés après dédoublonnage (PZ8)
- **Confiance moyenne** : 0.896
- **Temps de traitement** : ~2 min pour 52s de vidéo (CPU)
- **Sorties** : eport_final.json + eport_final.xlsx

> Développé pour la курсовая работа | Vision par Ordinateur & IA Multimodale