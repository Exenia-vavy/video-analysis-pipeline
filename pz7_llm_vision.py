import os
import cv2
import json
import time
import logging
import re
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONFIGURATION & LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Prompt strict en russe + format JSON imposé
RUSSIAN_PROMPT = """
Определи все видимые объекты на этом изображении. Верни ТОЛЬКО валидный JSON-массив.
Каждый объект должен быть словарём с ровно такими ключами:
- "class_name": строка (на русском или английском, например "человек", "машина", "собака")
- "confidence": число от 0.0 до 1.0
- "bbox": список из 4 целых чисел [x1, y1, x2, y2] в пикселях (приблизительно, если нужно)
Не добавляй объяснений, markdown или лишний текст. Если объектов нет, верни [].
"""

def setup_gemini(api_key: str):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def setup_qwen_openrouter(api_key: str):
    import requests
    return {"type": "qwen", "api_key": api_key, "session": requests.Session()}

def call_llm_russian(client, frame):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            if client.get("type") == "qwen":
                # Qwen via OpenRouter (format multimodal OpenAI-compatible)
                import base64, io
                buffer = io.BytesIO()
                img_pil.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                headers = {
                    "Authorization": f"Bearer {client['api_key']}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://local-pz7.local",
                    "X-Title": "PZ7-Coursework"
                }
                payload = {
                    "model": "qwen/qwen2.5-vl-72b-instruct",
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "text", "text": RUSSIAN_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300
                }
                resp = client["session"].post("https://openrouter.ai/api/v1/chat/completions", 
                                             json=payload, headers=headers)
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
            else:
                # Gemini
                raw = client.generate_content([RUSSIAN_PROMPT, img_pil]).text

            # Nettoyage robuste pour JSON cyrillique
            cleaned = re.sub(r'```(?:json)?\s*', '', raw).replace('```', '').strip()
            match = re.search(r'\[\s*\{.*?\}\s*\]', cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            time.sleep(1.5)
        except Exception as e:
            logging.warning(f"⚠️ Erreur API (tentative {attempt+1}) : {e}")
            time.sleep(3)
    return []

# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────
def run_llm_detection_ru(
    video_path: Path,
    output_dir: Path,
    backend: str = "qwen",  # "qwen" ou "gemini"
    api_key: str = "",
    sample_interval_sec: float = 5.0
) -> dict | None:
    if not video_path.exists():
        logging.error(f"📁 Vidéo introuvable : {video_path}")
        return None

    logging.info("🤖 Initialisation du backend LLM...")
    if backend.lower() == "qwen":
        if not api_key:
            logging.error("❌ Clé OpenRouter requise pour Qwen")
            return None
        client = setup_qwen_openrouter(api_key)
        logging.info("✅ Backend : Qwen-VL (via OpenRouter)")
    else:
        if not api_key:
            logging.error("❌ Clé Gemini requise")
            return None
        client = setup_gemini(api_key)
        logging.info("✅ Backend : Gemini 1.5 Flash")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error("❌ Impossible d'ouvrir la vidéo")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps * sample_interval_sec))

    detections_data = {
        "source_video": video_path.name,
        "model_used": f"{backend}_vl",
        "sample_interval_sec": sample_interval_sec,
        "frame_detections": []
    }

    frame_idx = 0
    logging.info(f"🎬 Analyse LLM (RU) : ~{total_frames/frame_step:.0f} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp = round(frame_idx / fps, 2)
            logging.info(f"🔍 Frame {frame_idx} (t={timestamp}s)...")

            objects_raw = call_llm_russian(client, frame)
            objects_normalized = []
            for obj in objects_raw:
                objects_normalized.append({
                    "class_id": -1,
                    "class_name": obj.get("class_name", "объект").lower(),
                    "confidence": float(obj.get("confidence", 0.5)),
                    "bbox": obj.get("bbox", [0, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))])
                })

            if objects_normalized:
                detections_data["frame_detections"].append({
                    "frame_id": frame_idx,
                    "timestamp_seconds": timestamp,
                    "objects": objects_normalized,
                    "object_count": len(objects_normalized)
                })
            time.sleep(1.2)  # Contrôle de coût/rate-limit

        frame_idx += 1

    cap.release()

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "llm_detections_ru.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections_data, f, indent=2, ensure_ascii=False)  # ← Garde le cyrillique lisible
        
    logging.info(f"💾 Résultats LLM (RU) sauvegardés : {json_path}")
    return detections_data

# ─────────────────────────────────────────────────────────────
# EXÉCUTION
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Choix du backend : "qwen" (recommandé pour RU) ou "gemini"
    BACKEND = "qwen"
    
    # Clés API (mieux via variables d'environnement)
    API_KEY = os.getenv("LLM_API_KEY", "")
    
    VIDEO_PATH = Path(r"C:\Users\Exenia\Desktop\videoss\video2.mp4")
    OUT_DIR = VIDEO_PATH.parent / "llm_pz7_ru"

    if not API_KEY:
        logging.error("❌ Définissez LLM_API_KEY (OpenRouter pour Qwen ou Google pour Gemini)")
    else:
        run_llm_detection_ru(VIDEO_PATH, OUT_DIR, backend=BACKEND, api_key=API_KEY, sample_interval_sec=5.0)