import argparse
import base64
import json
import time
from typing import Dict, Any, List

import cv2
import numpy as np
import requests

# YOLO (Ultralytics)
from ultralytics import YOLO


# -----------------------------
# Settings
# -----------------------------
OLLAMA_BASE = "http://localhost:11434"
CAPTION_MODEL = "minicpm-v"
REASON_MODEL = "llama3.2:latest"

# Motion gating
MOTION_DOWNSCALE = (160, 90)   # small for speed
MOTION_THRESH = 6.0            # typical: 4..12 (tune)
MOTION_SMOOTH = 0.6            # EMA smoothing

# YOLO cadence (run every N frames)
DETECT_EVERY_SEC = 0.4         # run YOLO about 2-3 times/sec
YOLO_CONF = 0.35               # detection confidence threshold
YOLO_IMGSZ = 416               # smaller=faster (320/416/512)

# Trigger rules
CAPTION_COOLDOWN_SEC = 2.0     # minimum time between captions
REFRESH_SEC = 8.0              # caption at least once every N sec (even if quiet)

# Caption speed settings
CAPTION_LONG_SIDE = 384
JPEG_QUALITY = 60
MAX_CAPTION_TOKENS = 40

# Which classes trigger captioning
# COCO: person=0, car=2, motorcycle=3, bus=5, truck=7
TRIGGER_CLASSES = {0, 2, 3, 5, 7}
CLASS_NAME_MAP = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# -----------------------------
# Ollama helpers
# -----------------------------
def jpeg_b64(frame_bgr: np.ndarray, long_side: int, quality: int) -> str:
    h, w = frame_bgr.shape[:2]
    scale = long_side / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def ollama_generate_vision(prompt: str, frame_bgr: np.ndarray) -> str:
    payload = {
        "model": CAPTION_MODEL,
        "prompt": prompt,
        "images": [jpeg_b64(frame_bgr, CAPTION_LONG_SIDE, JPEG_QUALITY)],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": MAX_CAPTION_TOKENS},
    }
    r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def ollama_chat(system: str, user: str) -> str:
    payload = {
        "model": REASON_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 700},
    }
    r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    return (r.json().get("message", {}).get("content") or "").strip()


# -----------------------------
# Motion score
# -----------------------------
def motion_score(prev_gray_small: np.ndarray, gray_small: np.ndarray) -> float:
    # mean absolute difference
    return float(cv2.absdiff(gray_small, prev_gray_small).mean())


# -----------------------------
# YOLO detection
# -----------------------------
def yolo_detect(yolo: YOLO, frame_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Returns:
      {
        "has_trigger": bool,
        "objects": [{"cls":int,"name":str,"conf":float,"xyxy":[x1,y1,x2,y2]}, ...]
      }
    """
    # Ultralytics expects BGR/np array fine
    res = yolo.predict(frame_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)[0]
    objs = []

    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        conf = boxes.conf.detach().cpu().numpy()
        xyxy = boxes.xyxy.detach().cpu().numpy()

        for c, cf, bb in zip(cls, conf, xyxy):
            if c in TRIGGER_CLASSES:
                objs.append({
                    "cls": int(c),
                    "name": CLASS_NAME_MAP.get(int(c), str(int(c))),
                    "conf": float(cf),
                    "xyxy": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                })

    return {
        "has_trigger": len(objs) > 0,
        "objects": objs,
    }


# -----------------------------
# Main pipeline
# -----------------------------
def run(video_path: str, out_json: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames > 0 else 0.0

    print(f"Video FPS: {fps:.2f} | frames: {total_frames} | duration: {duration:.2f}s")

    # YOLO tiny model
    print("Loading YOLO (yolov8n)...")
    yolo = YOLO("yolov8n.pt")

    det_every_n = max(1, int(round(DETECT_EVERY_SEC * fps)))
    print(f"YOLO will run every {det_every_n} frames (~{DETECT_EVERY_SEC}s)")

    caption_prompt = (
        "Describe the person's action in 6-10 words. "
        "Use one verb only from: walking, standing, bending, picking, placing, "
        "operating machine, using tool, inspecting. "
        "If unclear: unclear."
    )

    events: List[Dict[str, Any]] = []

    prev_gray_small = None
    motion_ema = 0.0

    last_caption_t = -1e9
    last_refresh_t = -1e9

    # cache last detections (updated every det_every_n frames)
    last_det = {"has_trigger": False, "objects": []}

    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = frame_idx / fps

        # motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, MOTION_DOWNSCALE, interpolation=cv2.INTER_AREA)

        if prev_gray_small is None:
            m = 0.0
        else:
            m = motion_score(prev_gray_small, gray_small)

        prev_gray_small = gray_small
        motion_ema = MOTION_SMOOTH * motion_ema + (1.0 - MOTION_SMOOTH) * m

        # yolo every N frames
        if frame_idx % det_every_n == 0:
            last_det = yolo_detect(yolo, frame)

        # trigger condition
        motion_trigger = (motion_ema >= MOTION_THRESH)
        obj_trigger = bool(last_det.get("has_trigger", False))

        refresh_due = (t_sec - last_refresh_t) >= REFRESH_SEC
        cooldown_ok = (t_sec - last_caption_t) >= CAPTION_COOLDOWN_SEC

        should_caption = cooldown_ok and ((motion_trigger and obj_trigger) or refresh_due)

        if should_caption:
            # caption with MiniCPM
            try:
                cap_text = ollama_generate_vision(caption_prompt, frame)
            except Exception as e:
                cap_text = f"[caption_error] {e}"

            ev = {
                "t": round(t_sec, 2),
                "motion": round(float(motion_ema), 3),
                "motion_trigger": bool(motion_trigger),
                "detected": last_det,  # includes bounding boxes for person/car/etc
                "caption": cap_text,
            }
            events.append(ev)

            print(f"[CAPTION] t={t_sec:.2f}s motion={motion_ema:.2f} "
                  f"motionTrig={motion_trigger} objTrig={obj_trigger} "
                  f"objs={len(last_det.get('objects', []))} -> {cap_text}")

            last_caption_t = t_sec
            last_refresh_t = t_sec

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0
    print(f"Finished scanning. Captions: {len(events)} | elapsed: {elapsed:.1f}s")

    # If no events were captured, force a fallback summary (first/mid/last)
    if len(events) == 0:
        print("No events captured; forcing 3 fallback captions (first/mid/last).")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()

        picks = [0, len(frames)//2, max(0, len(frames)-1)]
        for i in picks:
            t_sec = i / fps
            try:
                cap_text = ollama_generate_vision(caption_prompt, frames[i])
            except Exception as e:
                cap_text = f"[caption_error] {e}"
            events.append({"t": round(t_sec, 2), "motion": None, "detected": None, "caption": cap_text})

    # LLM summary (llama3.2)
    system = (
        "You are a CCTV video understanding assistant. "
        "You receive sparse event captions (primary evidence) plus detection context (person/car). "
        "Do not invent details not supported by captions. If uncertain, say so."
    )
    user = (
        "Analyze these time-ordered events and return:\n"
        "1) Overall summary (2-4 lines)\n"
        "2) Key events (3-8 bullets with timestamps)\n"
        "3) Any uncertainty / missing info\n\n"
        f"Events JSON:\n{json.dumps(events, indent=2)}"
    )
    summary = ollama_chat(system, user)

    result = {
        "video": video_path,
        "fps": round(float(fps), 3),
        "duration_sec": round(float(duration), 2),
        "det_every_n_frames": det_every_n,
        "motion_thresh": MOTION_THRESH,
        "refresh_sec": REFRESH_SEC,
        "caption_cooldown_sec": CAPTION_COOLDOWN_SEC,
        "events": events,
        "llm_summary": summary,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== SUMMARY ===")
    print(summary)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="hybrid_yolo_out.json")
    args = ap.parse_args()

    run(args.video, args.out)
