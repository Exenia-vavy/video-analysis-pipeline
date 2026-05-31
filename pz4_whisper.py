import subprocess
import json
import logging
from pathlib import Path
import whisper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def extract_audio(video_path: Path, output_wav: Path, sample_rate: int = 16000) -> bool:
    """Extrait l'audio d'une vidéo en WAV 16kHz mono via FFmpeg."""
    if not video_path.exists():
        logging.error(f"📁 Fichier vidéo introuvable : {video_path}")
        return False

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", str(sample_rate), "-ac", "1",
        "-f", "wav", str(output_wav)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"✅ Audio extrait : {output_wav}")
        return True
    except FileNotFoundError:
        logging.error("❌ FFmpeg non trouvé. Installez-le et ajoutez-le au PATH système.")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Erreur FFmpeg : {e.stderr.strip()}")
        return False

def transcribe_audio(audio_path: Path, model_size: str = "base", target_language: str = None) -> dict:
    """Transcrit un fichier audio avec Whisper et retourne une structure JSON prête pour ПЗ8."""
    logging.info(f"🎧 Chargement du modèle Whisper ({model_size})...")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        logging.error(f"❌ Échec du chargement du modèle Whisper : {e}")
        return {}

    logging.info("🔍 Transcription en cours (peut prendre quelques minutes)...")
    try:
        result = model.transcribe(str(audio_path), language=target_language, word_timestamps=False)
    except Exception as e:
        logging.error(f"❌ Erreur de transcription : {e}")
        return {}

    segments_data = []
    for seg in result.get("segments", []):
        segments_data.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })

    return {
        "source_audio": audio_path.name,
        "model_used": model_size,
        "detected_language": result.get("language", "unknown"),
        "segments": segments_data,
        "full_transcription": result["text"].strip()
    }

def process_audio_pipeline(video_path: Path, output_dir: Path, model_size: str = "base", language: str = None) -> dict | None:
    """Orchestre l'extraction et la transcription. Retourne le dictionnaire JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "extracted_audio.wav"
    json_path = output_dir / "audio_transcription.json"

    if not extract_audio(video_path, wav_path):
        return None

    transcription = transcribe_audio(wav_path, model_size, language)
    if not transcription:
        return None

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcription, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 Transcription sauvegardée : {json_path}")
    return transcription

# ─────────────────────────────────────────────────────────────
# EXÉCUTION
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    VIDEO_PATH = Path(r"C:\Users\Exenia\Desktop\videoss\video2.mp4")
    OUT_DIR = VIDEO_PATH.parent / "audio_pz4"

    logging.info("🚀 Lancement du pipeline audio (ПЗ4)...")
    # language=None → auto-détection. Mettre "ru" ou "en" si connu.
    process_audio_pipeline(VIDEO_PATH, OUT_DIR, model_size="base", language=None)