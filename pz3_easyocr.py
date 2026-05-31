import cv2
import easyocr
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─────────────────────────────────────────────────────────────
# INITIALISATION OPTIMISÉE (chargement unique du modèle)
# ─────────────────────────────────────────────────────────────
_OCR_READER = None

def get_ocr_reader(languages: List[str] = ['ru', 'en'], use_gpu: bool = False) -> easyocr.Reader:
    global _OCR_READER
    if _OCR_READER is None:
        logging.info("🔍 Chargement du modèle EasyOCR (peut prendre 10-20s)...")
        _OCR_READER = easyocr.Reader(languages, gpu=use_gpu, verbose=False)
        logging.info("✅ Modèle EasyOCR prêt.")
    return _OCR_READER

def clean_text(text: str) -> str:
    """Nettoie le texte : conserve lettres cyrilliques/latines, chiffres et ponctuation."""
    # Regex compatible Python standard pour RU/EN + ponctuation courante
    cleaned = re.sub(r'[^\w\sа-яА-ЯёЁ.,!?;:\-()"]', '', text)
    return cleaned.strip()

def run_ocr_on_frames(
    frames_dir: Path,
    metadata_path: Path = None,
    confidence_threshold: float = 0.35,
    languages: List[str] = ['ru', 'en']
) -> Dict[str, Any]:
    """
    Exécute l'OCR sur un dossier de frames (idéalement sorties de ПЗ2).
    Retourne un dictionnaire structuré prêt pour ПЗ8 (dédoublonnage).
    """
    reader = get_ocr_reader(languages)
    ocr_results: Dict[str, Any] = {
        "source_directory": frames_dir.name,
        "confidence_threshold": confidence_threshold,
        "frame_results": []
    }

    # Chargement optionnel des métadonnées ПЗ2 pour mapper timestamps
    frame_meta = {}
    if metadata_path and metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        for img_info in meta.get("extracted_images", []):
            frame_meta[img_info["filename"]] = img_info

    images = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not images:
        logging.warning("📁 Aucune image trouvée dans le dossier frames.")
        return ocr_results

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # OCR EasyOCR
        detections = reader.readtext(img)
        valid_texts = []
        for bbox, text, conf in detections:
            if conf >= confidence_threshold:
                cleaned = clean_text(text)
                if cleaned and len(cleaned) > 1:  # Ignore les caractères seuls/bruit
                    valid_texts.append({
                        "text": cleaned,
                        "confidence": round(conf, 3),
                        "bbox": [[int(c) for c in point] for point in bbox]
                    })

        if valid_texts:
            meta_info = frame_meta.get(img_path.name, {})
            ocr_results["frame_results"].append({
                "image_file": img_path.name,
                "timestamp_seconds": meta_info.get("timestamp_seconds"),
                "frame_id": meta_info.get("id"),
                "texts": valid_texts
            })
            logging.info(f"📝 [{img_path.name}] {len(valid_texts)} texte(s) détecté(s)")

    return ocr_results


# ─────────────────────────────────────────────────────────────
# INTERFACE DIRECTE (vidéo brute) - Pour tests rapides
# ─────────────────────────────────────────────────────────────
def run_ocr_on_video(video_path: Path, output_dir: Path, step_sec: int = 5, languages=['ru', 'en']):
    """Fallback : scanne directement une vidéo toutes les X secondes (comportement initial)."""
    reader = get_ocr_reader(languages)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"⚠️ Impossible d'ouvrir {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    step_frames = int(step_sec * fps)

    results = {"source_video": video_path.name, "frame_results": []}
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: break

        detections = reader.readtext(frame)
        valid_texts = [{"text": clean_text(t), "confidence": round(c, 3)} 
                       for _, t, c in detections if c >= 0.35 and len(clean_text(t)) > 1]

        if valid_texts:
            results["frame_results"].append({
                "timestamp_seconds": round(frame_idx / fps, 2),
                "texts": valid_texts
            })

        frame_idx += step_frames

    cap.release()
    
    # Sauvegarde JSON explicite
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ocr_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 Résultats OCR sauvegardés : {json_path}")
    return results


# ─────────────────────────────────────────────────────────────
# EXÉCUTION
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    # MODE 1 : Recommandé pour le pipeline (utilise les frames de ПЗ2)
    PZ2_FRAMES_DIR = Path(r"C:\Users\Exenia\Desktop\videoss\frames_video2_1fps")
    PZ2_METADATA = PZ2_FRAMES_DIR / "metadata.json"
    
    if PZ2_FRAMES_DIR.exists():
        ocr_data = run_ocr_on_frames(PZ2_FRAMES_DIR, PZ2_METADATA)
        out_path = PZ2_FRAMES_DIR / "ocr_results.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_data, f, indent=2, ensure_ascii=False)
        logging.info(f"✅ OCR terminé sur frames ПЗ2 → {out_path}")
    else:
        # MODE 2 : Fallback vidéo directe (comportement initial)
        video = Path(r"C:\Users\Exenia\Desktop\videoss\video2.mp4")
        out_dir = video.parent / "ocr_scan"
        logging.info("🔄 Fallback : scan direct de la vidéo...")
        run_ocr_on_video(video, out_dir, step_sec=5)