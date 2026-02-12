import argparse, base64, json, time
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import requests
from ultralytics import YOLO

OLLAMA_BASE = "http://localhost:11434"
CAPTION_MODEL = "minicpm-v"
REASON_MODEL  = "llama3.2:latest"

# YOLO settings
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF  = 0.35
YOLO_IMGSZ = 416
PERSON_CLS = 0

# Run detection every N frames (speed)
DETECT_EVERY_SEC = 0.4

# Human-action gate (motion inside person bbox)
ROI_DOWNSCALE_W = 96
ROI_MOTION_THRESH = 4.5      # tune: 3..8
ROI_MOTION_SMOOTH = 0.6      # EMA

# Caption throttles
CAPTION_COOLDOWN_SEC = 2.0
MIN_ACTION_DURATION_SEC = 0.6   # require action persists this long before caption
REFRESH_WHILE_ACTIVE_SEC = 4.0  # while active, caption at most once per N sec

# Caption speed
CAPTION_LONG_SIDE = 384
JPEG_QUALITY = 60
MAX_CAPTION_TOKENS = 40


def jpeg_b64(frame_bgr: np.ndarray, long_side=384, quality=60) -> str:
    h, w = frame_bgr.shape[:2]
    scale = long_side / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    frame_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
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


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def roi_motion(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    # mean absolute difference
    return float(cv2.absdiff(gray, prev_gray).mean())


def detect_persons(yolo: YOLO, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    res = yolo.predict(frame_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)[0]
    persons = []
    if res.boxes is None or len(res.boxes) == 0:
        return persons
    cls = res.boxes.cls.detach().cpu().numpy().astype(int)
    conf = res.boxes.conf.detach().cpu().numpy()
    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    for c, cf, bb in zip(cls, conf, xyxy):
        if int(c) == PERSON_CLS:
            persons.append({"conf": float(cf), "xyxy": bb.tolist()})
    # take biggest person first (most relevant)
    persons.sort(key=lambda p: (p["xyxy"][2]-p["xyxy"][0])*(p["xyxy"][3]-p["xyxy"][1]), reverse=True)
    return persons


def crop_person_roi_gray(frame_bgr: np.ndarray, person_xyxy: List[float]) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = person_xyxy
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
    roi = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # downscale width to ROI_DOWNSCALE_W, keep aspect
    rh, rw = gray.shape[:2]
    if rw > 0:
        scale = ROI_DOWNSCALE_W / rw
        nw = ROI_DOWNSCALE_W
        nh = max(1, int(rh * scale))
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    return gray


def run(video_path: str, out_json: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0.0
    print(f"FPS={fps:.2f} frames={total_frames} duration={duration:.2f}s")

    yolo = YOLO(YOLO_MODEL)
    det_every_n = max(1, int(round(DETECT_EVERY_SEC * fps)))
    print(f"YOLO every {det_every_n} frames (~{DETECT_EVERY_SEC}s)")

    caption_prompt = (
        "Describe the person's action in 6-10 words. "
        "Use one verb only from: walking, standing, bending, picking, placing, "
        "operating machine, using tool, inspecting. "
        "If unclear: unclear."
    )

    events = []

    last_caption_t = -1e9
    last_active_caption_t = -1e9

    # person ROI motion state
    prev_roi_gray: Optional[np.ndarray] = None
    motion_ema = 0.0

    # detection cache
    last_persons: List[Dict[str, Any]] = []
    person_present = False

    # action persistence
    action_start_t: Optional[float] = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        # Update person detection periodically
        if frame_idx % det_every_n == 0:
            last_persons = detect_persons(yolo, frame)
            person_present = len(last_persons) > 0

            # reset ROI motion history if no person
            if not person_present:
                prev_roi_gray = None
                motion_ema = 0.0
                action_start_t = None

        human_action = False
        roi_motion_val = 0.0

        if person_present:
            # Use the biggest person bbox
            roi_gray = crop_person_roi_gray(frame, last_persons[0]["xyxy"])

            if prev_roi_gray is None or prev_roi_gray.shape != roi_gray.shape:
                roi_motion_val = 0.0
            else:
                roi_motion_val = roi_motion(prev_roi_gray, roi_gray)

            prev_roi_gray = roi_gray

            # Smooth motion to avoid flicker
            motion_ema = ROI_MOTION_SMOOTH * motion_ema + (1 - ROI_MOTION_SMOOTH) * roi_motion_val

            # Action gate: motion inside person ROI
            if motion_ema >= ROI_MOTION_THRESH:
                human_action = True

        # Persist action for MIN_ACTION_DURATION_SEC
        if human_action:
            if action_start_t is None:
                action_start_t = t
        else:
            action_start_t = None

        action_persisted = (action_start_t is not None) and ((t - action_start_t) >= MIN_ACTION_DURATION_SEC)

        # Caption policy:
        # - Only caption if action persisted
        # - Cooldown across all captions
        # - While active, do periodic captions (REFRESH_WHILE_ACTIVE_SEC)
        cooldown_ok = (t - last_caption_t) >= CAPTION_COOLDOWN_SEC
        active_refresh_due = (t - last_active_caption_t) >= REFRESH_WHILE_ACTIVE_SEC

        should_caption = action_persisted and cooldown_ok and active_refresh_due

        if should_caption:
            try:
                cap_text = ollama_generate_vision(caption_prompt, frame)
            except Exception as e:
                cap_text = f"[caption_error] {e}"

            ev = {
                "t": round(t, 2),
                "person": {
                    "conf": round(last_persons[0]["conf"], 3),
                    "xyxy": [round(v, 1) for v in last_persons[0]["xyxy"]],
                },
                "roi_motion_ema": round(float(motion_ema), 3),
                "caption": cap_text,
            }
            events.append(ev)

            print(f"[CAPTION] t={t:.2f}s motion={motion_ema:.2f} -> {cap_text}")

            last_caption_t = t
            last_active_caption_t = t

        frame_idx += 1

    cap.release()
    print(f"Done. Captions produced: {len(events)}")

    # If zero events, do nothing-or fallback: here we keep zero and let LLM say "no action"
    system = (
        "You are a CCTV video understanding assistant. "
        "You receive captions ONLY when human action was detected (primary evidence). "
        "If the event list is empty, conclude that there was no significant human action."
    )
    user = (
        "Summarize what is happening in the video using ONLY the events.\n"
        "Return:\n"
        "1) Overall summary\n"
        "2) Key events (bullets with timestamps)\n"
        "3) Uncertainty / missing info\n\n"
        f"Events JSON:\n{json.dumps(events, indent=2)}"
    )
    summary = ollama_chat(system, user)

    out = {
        "video": video_path,
        "fps": round(float(fps), 3),
        "duration_sec": round(float(duration), 2),
        "yolo_every_n_frames": det_every_n,
        "roi_motion_thresh": ROI_MOTION_THRESH,
        "min_action_duration_sec": MIN_ACTION_DURATION_SEC,
        "caption_cooldown_sec": CAPTION_COOLDOWN_SEC,
        "active_refresh_sec": REFRESH_WHILE_ACTIVE_SEC,
        "events": events,
        "llm_summary": summary,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== SUMMARY ===")
    print(summary)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="human_action_out.json")
    args = ap.parse_args()
    run(args.video, args.out)
