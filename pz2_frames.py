import cv2
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIGURATION & LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def extract_frames(video_path: Path, output_dir: Path, target_fps: float = 1.0) -> dict | None:
    """
    Extrait une image toutes les X frames pour atteindre target_fps.
    Retourne un dictionnaire de métadonnées prêt pour les ПЗ suivants.
    """
    if not video_path.exists():
        logging.error(f"📁 Fichier vidéo introuvable : {video_path}")
        return None
    if video_path.stat().st_size == 0:
        logging.error("📁 Fichier vidéo vide.")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"⚠️ Impossible d'ouvrir la vidéo avec OpenCV : {video_path.name}")
        return None

    # Récupération robuste du FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logging.warning("FPS non détecté, valeur par défaut : 30.0")
        fps = 30.0

    frame_interval = max(1, round(fps / target_fps))
    logging.info(f"🎬 FPS original : {fps:.2f} | Intervalle : {frame_interval} frames (~{target_fps} img/sec)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Structure JSON standardisée pour le pipeline
    metadata = {
        "source_video": video_path.name,
        "original_fps": fps,
        "target_fps": target_fps,
        "frame_interval": frame_interval,
        "total_frames_extracted": 0,
        "extracted_images": []
    }

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            img_name = f"frame_{saved_count:04d}.png"
            img_path = output_dir / img_name
            cv2.imwrite(str(img_path), frame)

            timestamp = round(frame_count / fps, 2)
            metadata["extracted_images"].append({
                "id": saved_count,
                "filename": img_name,
                "timestamp_seconds": timestamp,
                "llm_detection_results": []  # Sera rempli par ПЗ7 / ПЗ5 / ПЗ6
            })
            saved_count += 1

        frame_count += 1

    cap.release()
    metadata["total_frames_extracted"] = saved_count

    # Sauvegarde JSON
    json_path = output_dir / "metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logging.info(f"✅ Extraction terminée : {saved_count} images sauvegardées")
    logging.info(f"📄 Métadonnées : {json_path}")
    return metadata


# ─────────────────────────────────────────────────────────────
# INTERFACE INTERACTIVE (UNIQUEMENT POUR TESTS MANUELS)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Forcer UTF-8 sur Windows Console
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    VIDEO_DIR = Path(r"C:\Users\Exenia\Desktop\videoss")
    if not VIDEO_DIR.exists():
        logging.error("Dossier vidéo introuvable. Modifiez VIDEO_DIR dans le script.")
        sys.exit(1)

    valid_ext = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
    videos = sorted([f for f in VIDEO_DIR.iterdir() if f.suffix.lower() in valid_ext])

    if not videos:
        logging.warning("Aucune vidéo trouvée dans le dossier.")
        sys.exit(0)

    # Sélection simple
    if len(videos) == 1:
        chosen = videos[0]
    else:
        print("\n📹 Vidéos disponibles :")
        for i, v in enumerate(videos, 1):
            print(f"  {i}. {v.name}")
        while True:
            try:
                idx = int(input(f"\nChoisissez un numéro (1-{len(videos)}) : ")) - 1
                if 0 <= idx < len(videos):
                    chosen = videos[idx]
                    break
                print(f"Numéro invalide. Entrez 1 à {len(videos)}.")
            except ValueError:
                print("Veuillez entrer un entier.")

    OUTPUT_DIR = VIDEO_DIR / f"frames_{chosen.stem}_1fps"
    extract_frames(chosen, OUTPUT_DIR, target_fps=1.0)