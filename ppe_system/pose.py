# run_clip.py
import json
import re
import yaml
import numpy as np

from ppe_system.video_io import iter_sampled_frames
from ppe_system.gating import yolo_person_gate, motion_gate


# ----------------------------
# Ollama (llama3.2) utilities
# ----------------------------
def _ollama_chat(base_url: str, model: str, messages, timeout_sec: int = 180) -> str:
    import requests

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    return r.json()["message"]["content"]


def extract_json_object(text: str) -> dict:
    # strip markdown fences if any
    text = text.replace("```json", "").replace("```", "").strip()

    # get from first '{' to end
    m = re.search(r"\{.*", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON start '{' found in LLM output.")

    candidate = m.group(0)

    # auto-close missing braces
    open_braces = candidate.count("{")
    close_braces = candidate.count("}")
    if close_braces < open_braces:
        candidate = candidate + ("}" * (open_braces - close_braces))

    # remove trailing commas
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)

    return json.loads(candidate)


def _safe_float(x, default=0.5) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


# ----------------------------
# Pose + features
# ----------------------------
LANDMARKS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

def _angle(a, b, c) -> float | None:
    """Angle at point b formed by a-b-c (degrees)."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return None
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    if denom <= 0:
        return None
    cosang = float(np.dot(ba, bc) / denom)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _dist(a, b) -> float | None:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return None
    return float(np.linalg.norm(a - b))


def extract_pose_keypoints(timed_frames, model_complexity: int = 1):
    """
    Returns list of: {"t": float, "kp": {name: [x,y,z,vis]}}
    x,y are normalized [0..1] wrt image; z is relative; vis in [0..1]
    """
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError(
            "mediapipe is required for pose approach.\n"
            "Install it: pip install mediapipe\n"
            f"Original error: {e}"
        )

    mp_pose = mp.solutions.pose

    out = []
    with mp_pose.Pose(static_image_mode=True, model_complexity=model_complexity) as pose:
        for t, frame_bgr in timed_frames:
            img_rgb = frame_bgr[:, :, ::-1]
            res = pose.process(img_rgb)

            kp = {}
            if res.pose_landmarks:
                # MediaPipe Pose landmark indices
                lm = res.pose_landmarks.landmark

                # Map the subset we care about
                mapping = {
                    "nose": 0,
                    "left_shoulder": 11, "right_shoulder": 12,
                    "left_elbow": 13, "right_elbow": 14,
                    "left_wrist": 15, "right_wrist": 16,
                    "left_hip": 23, "right_hip": 24,
                    "left_knee": 25, "right_knee": 26,
                    "left_ankle": 27, "right_ankle": 28,
                }

                for name, idx in mapping.items():
                    p = lm[idx]
                    kp[name] = [float(p.x), float(p.y), float(p.z), float(p.visibility)]

            out.append({"t": float(t), "kp": kp})
    return out


def compute_pose_features(timed_kp):
    """
    Build compact per-time features:
    - angles: elbows, knees
    - distances normalized by torso length
    - velocities (wrist) between consecutive frames (normalized by torso length)
    """
    feats = []

    prev = None
    prev_t = None

    for item in timed_kp:
        t = item["t"]
        kp = item["kp"]

        def get_xyz(name):
            v = kp.get(name)
            if not v:
                return [np.nan, np.nan, np.nan]
            return v[:3]

        # points
        LS, RS = get_xyz("left_shoulder"), get_xyz("right_shoulder")
        LE, RE = get_xyz("left_elbow"), get_xyz("right_elbow")
        LW, RW = get_xyz("left_wrist"), get_xyz("right_wrist")
        LH, RH = get_xyz("left_hip"), get_xyz("right_hip")
        LK, RK = get_xyz("left_knee"), get_xyz("right_knee")
        LA, RA = get_xyz("left_ankle"), get_xyz("right_ankle")
        NOSE = get_xyz("nose")

        # torso length for normalization
        torso = _dist(LS, LH)
        if torso is None or torso <= 1e-6:
            torso = 1.0

        angles = {
            "left_elbow": _angle(LS, LE, LW),
            "right_elbow": _angle(RS, RE, RW),
            "left_knee": _angle(LH, LK, LA),
            "right_knee": _angle(RH, RK, RA),
        }

        distances = {
            "hands_apart": (_dist(LW, RW) / torso) if _dist(LW, RW) is not None else None,
            "left_wrist_to_nose": (_dist(LW, NOSE) / torso) if _dist(LW, NOSE) is not None else None,
            "right_wrist_to_nose": (_dist(RW, NOSE) / torso) if _dist(RW, NOSE) is not None else None,
            "left_wrist_to_left_hip": (_dist(LW, LH) / torso) if _dist(LW, LH) is not None else None,
            "right_wrist_to_right_hip": (_dist(RW, RH) / torso) if _dist(RW, RH) is not None else None,
        }

        # confidence: average visibility over available points
        vis = []
        for n in LANDMARKS:
            if n in kp:
                vis.append(float(kp[n][3]))
        pose_conf = float(np.mean(vis)) if vis else 0.0

        velocities = {"left_wrist": None, "right_wrist": None}
        if prev is not None and prev_t is not None:
            dt = max(1e-3, t - prev_t)

            def vnorm(curr_xyz, prev_xyz):
                if np.any(np.isnan(curr_xyz)) or np.any(np.isnan(prev_xyz)):
                    return None
                return float(np.linalg.norm(np.array(curr_xyz) - np.array(prev_xyz)) / dt) / torso

            velocities["left_wrist"] = vnorm(LW, prev["LW"])
            velocities["right_wrist"] = vnorm(RW, prev["RW"])

        feats.append({
            "t": round(t, 3),
            "pose_confidence": round(pose_conf, 3),
            "angles": angles,
            "distances": distances,
            "velocities": velocities,
        })

        prev = {"LW": LW, "RW": RW}
        prev_t = t

    return feats


def understand_action_from_pose_features(pose_features, allowed_actions, llm_cfg):
    """
    Use llama3.2 to infer action from pose-derived temporal features.
    Returns dict: action, confidence, summary, key_events
    """
    system = (
        "You are a human action analyst for CCTV/factory videos.\n"
        "You will receive time-ordered pose-derived features per timestamp:\n"
        "- angles (deg): elbows/knees\n"
        "- distances: normalized by torso length\n"
        "- velocities: normalized units per second (higher = faster motion)\n\n"
        "Your task: infer the most likely action.\n\n"
        "Return ONLY valid JSON (no markdown, no extra text) with schema:\n"
        "{\n"
        '  "action": "ONE_OF_ALLOWED_ACTIONS",\n'
        '  "confidence": 0.0,\n'
        '  "summary": "1-2 sentences grounded in features",\n'
        '  "key_events": [{"time":"t=4.2s","event":"..."}]\n'
        "}\n\n"
        "Rules:\n"
        "- action must be one of allowed_actions; else use UNKNOWN.\n"
        "- confidence is 0..1.\n"
        "- key_events must cite timestamps from input.\n"
        "- If pose_confidence is low or motion ambiguous, use UNKNOWN.\n"
    )

    user = {
        "allowed_actions": allowed_actions,
        "pose_features": pose_features,
        "notes": (
            "Interpretation hints:\n"
            "- walking: repeated leg angle changes + steady hip/ankle motion.\n"
            "- picking/placing: wrist velocity spike + elbow angle change + wrist moves toward hip/table region.\n"
            "- idle: very low wrist velocities and stable angles.\n"
            "- inspecting: small wrist movement + head/upper-body slightly changes; slower than picking.\n"
        )
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)}
    ]

    text = _ollama_chat(llm_cfg["base_url"], llm_cfg["model"], messages, timeout_sec=llm_cfg.get("timeout_sec", 180))

    try:
        data = extract_json_object(text)
    except Exception:
        print("\n--- RAW LLM OUTPUT START ---")
        print(text)
        print("--- RAW LLM OUTPUT END ---\n")
        raise

    action = str(data.get("action", "UNKNOWN")).strip()
    if action not in allowed_actions:
        action = "UNKNOWN"

    confidence = _safe_float(data.get("confidence", 0.5), default=0.5)
    summary = str(data.get("summary", "")).strip()

    key_events = data.get("key_events", [])
    if not isinstance(key_events, list):
        key_events = []
    cleaned = []
    for ev in key_events:
        if isinstance(ev, dict) and "time" in ev and "event" in ev:
            cleaned.append({"time": str(ev["time"]), "event": str(ev["event"])})

    return {"action": action, "confidence": confidence, "summary": summary, "key_events": cleaned}


# ----------------------------
# Main
# ----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a small clip video file")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", default="pose", choices=["pose", "caption"], help="pose=keypoints->LLM, caption=MiniCPM captions->LLM")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    # 1) Extract sampled frames
    timed_frames = []
    for t, frame in iter_sampled_frames(
        args.video,
        sample_fps=cfg["video"]["sample_fps"],
        resize_width=cfg["video"]["resize_width"],
        max_frames=cfg["video"]["max_frames"],
    ):
        timed_frames.append((t, frame))

    if not timed_frames:
        print("No frames extracted.")
        return

    # 2) Optional clip-level gating (kept as-is)
    if cfg.get("gating", {}).get("enabled", False):
        method = cfg["gating"].get("method", "motion")
        if method == "yolo":
            ok = yolo_person_gate(
                timed_frames,
                model_name=cfg["gating"]["yolo_model"],
                person_conf=cfg["gating"]["person_conf"],
                min_person_frames=cfg["gating"]["min_person_frames"],
            )
        else:
            frames = [f for _, f in timed_frames]
            ok = motion_gate(frames, thresh=cfg["gating"].get("motion_thresh", 0.02))

        if not ok:
            print("Gate: No meaningful activity detected. Skipping.")
            return

    if args.mode == "caption":
        # Old path (caption model) kept for comparison
        from ppe_system.captioner_llava_ollama import OllamaLlavaCaptioner
        from ppe_system.llm_reasoner import understand_video_from_captions

        captioner = OllamaLlavaCaptioner(
            model="minicpm-v",
            base_url=cfg.get("caption", {}).get("ollama_base_url", "http://localhost:11434"),
            prompt=cfg.get("caption", {}).get("prompt"),
            resize_long_side=cfg.get("caption", {}).get("resize_long_side", 320),
            jpeg_quality=cfg.get("caption", {}).get("jpeg_quality", 85),
            timeout_sec=cfg.get("caption", {}).get("timeout_sec", 120),
        )
        clip_caption = captioner.caption(timed_frames)

        print("\n=== CLIP CAPTION (MiniCPM-V) ===")
        print(f"{clip_caption.start_sec:.2f}s → {clip_caption.end_sec:.2f}s")
        print(clip_caption.caption)

        if cfg.get("llm", {}).get("enabled", True):
            res = understand_video_from_captions(
                clip_caption,
                allowed_actions=cfg["actions"]["allowed"],
                llm_cfg=cfg["llm"]["ollama"],
            )
            print("\n=== LLM VIDEO UNDERSTANDING ===")
            print(f"Action: {res.action} (conf={res.confidence:.2f})")
            print(f"Summary: {res.summary}")
            if res.key_events:
                print("Key events:")
                for e in res.key_events:
                    print(f" - {e['time']}: {e['event']}")
        return

    # ----------------------------
    # New path: Pose -> Features -> LLM
    # ----------------------------
    pose_cfg = cfg.get("pose", {})
    model_complexity = int(pose_cfg.get("model_complexity", 1))
    min_pose_conf = float(pose_cfg.get("min_pose_confidence", 0.25))
    max_points = int(pose_cfg.get("max_points_to_llm", 60))  # keep tokens low

    print(f"Pose mode: extracting keypoints (model_complexity={model_complexity}) ...")
    timed_kp = extract_pose_keypoints(timed_frames, model_complexity=model_complexity)
    pose_features = compute_pose_features(timed_kp)

    # Filter out very low-confidence points (optional)
    filtered = [x for x in pose_features if x["pose_confidence"] >= min_pose_conf]
    if not filtered:
        print("Pose confidence too low across frames. Cannot infer action.")
        return

    # Downsample for LLM token budget
    if len(filtered) > max_points:
        step = int(np.ceil(len(filtered) / max_points))
        filtered = filtered[::step]

    print(f"Pose features sent to LLM: {len(filtered)} points (from {len(pose_features)} total).")

    if cfg.get("llm", {}).get("enabled", True):
        llm_cfg = cfg["llm"]["ollama"]
        allowed_actions = cfg["actions"]["allowed"]
        out = understand_action_from_pose_features(filtered, allowed_actions, llm_cfg)

        print("\n=== LLM POSE-BASED ACTION UNDERSTANDING ===")
        print(f"Action: {out['action']} (conf={out['confidence']:.2f})")
        print(f"Summary: {out['summary']}")
        if out["key_events"]:
            print("Key events:")
            for e in out["key_events"]:
                print(f" - {e['time']}: {e['event']}")
    else:
        print("\nLLM disabled in config.yaml (llm.enabled: false).")


if __name__ == "__main__":
    main()
