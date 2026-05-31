import subprocess
import sys
import time
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Liste des scripts à exécuter dans l'ordre
PIPELINE_SCRIPTS = [
    "pz2_frames.py",
    "pz3_easyocr.py",
    "pz4_whisper.py",
    "pz5_yolo.py",
    "pz6_resnet.py",
    "pz7_llm_vision.py",
    "pz8_report.py"
]

def run_script(script_name: str) -> bool:
    """Exécute un script Python et retourne True si succès, False sinon."""
    script_path = Path(script_name)
    if not script_path.exists():
        logger.error(f" Script introuvable : {script_name}")
        return False

    logger.info(f"▶ Démarrage de {script_name}...")
    start_time = time.time()

    try:
        # Lance le script avec le même interpréteur Python
        # capture_output=False permet de voir les logs en temps réel dans la console
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        elapsed = time.time() - start_time
        logger.info(f"✅ {script_name} terminé en {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Échec de {script_name} (code retour {e.returncode})")
        return False
    except Exception as e:
        logger.error(f" Erreur inattendue avec {script_name} : {e}")
        return False

def main():
    logger.info("🚀 Lancement du pipeline multimodal complet...")
    logger.info("=" * 60)

    for script in PIPELINE_SCRIPTS:
        success = run_script(script)
        if not success:
            logger.critical("🛑 Pipeline interrompu. Vérifiez les logs ci-dessus.")
            sys.exit(1)
        logger.info("-" * 60)

    logger.info(" PIPELINE TERMINÉ AVEC SUCCÈS !")
    logger.info("📁 Résultats disponibles dans : output/final_report/")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()