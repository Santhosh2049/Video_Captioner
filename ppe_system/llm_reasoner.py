import json
import re
import requests
from .schemas import ClipCaption, VideoUnderstanding
import os
from dotenv import load_dotenv
load_dotenv()


def _ollama_chat(base_url: str, model: str, messages):
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]

def _openai_response(model: str, system_text: str, user_text: str) -> str:
    from openai import OpenAI
    client = OpenAI()

    r = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
    )
    # output_text is the simplest way to get the final combined text
    return r.output_text


def _safe_float(x, default=0.5) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)

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

def understand_video_from_captions(
    clip: ClipCaption,
    allowed_actions: list[str],
    llm_cfg: dict,
) -> VideoUnderstanding:

    system = (
        "You are a video analyst. You will be given timestamped frame captions from a short video.\n"
        "Your job: infer what is actually happening.\n\n"
        "Return ONLY valid JSON (no markdown, no extra text) with this schema:\n"
        "{\n"
        '  "action": "ONE_OF_ALLOWED_ACTIONS",\n'
        '  "confidence": 0.0,\n'
        '  "summary": "1-2 sentences describing what happens in the video",\n'
        '  "key_events": [\n'
        '     {"time": "t=0.0s", "event": "..."},\n'
        '     {"time": "t=2.4s", "event": "..."}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- confidence must be a number between 0 and 1.\n"
        "- key_events should be short, grounded in the given captions.\n"
        "- If uncertain, choose action=UNKNOWN and explain in summary.\n"
    )

    user = {
        "allowed_actions": allowed_actions,
        "clip_time": {"start_sec": clip.start_sec, "end_sec": clip.end_sec},
        "caption_timeline": clip.caption
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)}
    ]

    provider = llm_cfg.get("provider", "ollama")

    if provider == "openai":
        text = _openai_response(
            model=llm_cfg["openai"]["model"],
            system_text=system,
            user_text=json.dumps(user),
        )
    else:
        text = _ollama_chat(
            llm_cfg["ollama"]["base_url"],
            llm_cfg["ollama"]["model"],
            messages
        )


    try:
        data = extract_json_object(text)
    except Exception:
        print("\n--- RAW LLM OUTPUT START ---")
        print(text)
        print("--- RAW LLM OUTPUT END ---\n")
        raise

    action = data.get("action", "UNKNOWN")
    if action not in allowed_actions:
        action = "UNKNOWN"

    confidence = _safe_float(data.get("confidence", 0.5), default=0.5)
    summary = str(data.get("summary", "")).strip()
    key_events = data.get("key_events", [])
    if not isinstance(key_events, list):
        key_events = []

    # sanitize events
    cleaned_events = []
    for ev in key_events:
        if isinstance(ev, dict) and "time" in ev and "event" in ev:
            cleaned_events.append({"time": str(ev["time"]), "event": str(ev["event"])})

    return VideoUnderstanding(
        action=action,
        confidence=confidence,
        summary=summary,
        key_events=cleaned_events
    )
