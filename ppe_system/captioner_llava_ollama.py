from typing import List, Tuple
import base64
import io
import requests
import numpy as np
from PIL import Image

from .schemas import FrameEvidence, ClipCaption


class OllamaLlavaCaptioner:
    """
    Captions each frame by calling Ollama's /api/chat with LLaVA.
    Works offline if Ollama is running locally.
    """

    def __init__(
        self,
        model: str = "minicpm-v",
        base_url: str = "http://localhost:11434",
        prompt: str | None = None,
        resize_long_side: int = 256,
        jpeg_quality: int = 85,
        timeout_sec: int = 180,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.resize_long_side = int(resize_long_side)
        self.jpeg_quality = int(jpeg_quality)
        self.timeout_sec = int(timeout_sec)

        # This prompt style works well for CCTV / industrial scenes
        self.prompt = prompt or (
            "Describe key events happening across all frame in short sentence. "
            "Be concrete: mention person actions, "
            "and scene if visible. "
            "Avoid guessing location/country. No extra commentary."
        )

    def _bgr_to_pil(self, frame_bgr: np.ndarray) -> Image.Image:
        rgb = frame_bgr[:, :, ::-1]
        return Image.fromarray(rgb)

    def _resize_pil(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        m = max(w, h)
        if m <= self.resize_long_side:
            return img
        scale = self.resize_long_side / float(m)
        nw, nh = int(w * scale), int(h * scale)
        return img.resize((nw, nh))

    def _pil_to_base64_jpeg(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _caption_one(self, img_b64: str) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": self.prompt,
                    "images": [img_b64],
                }
            ],
        }
        r = requests.post(url, json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        # Ollama chat response format: {"message": {"content": "..."}}
        text = data.get("message", {}).get("content", "")
        return (text or "").strip()

    def caption(self, timed_frames: List[Tuple[float, np.ndarray]]) -> ClipCaption:
        if not timed_frames:
            return ClipCaption(0.0, 0.0, "Empty clip.", [])

        timed_frames = sorted(timed_frames, key=lambda x: x[0])
        evidence: List[FrameEvidence] = []
        caps: List[Tuple[float, str]] = []

        for t, frame in timed_frames:
            pil = self._bgr_to_pil(frame)
            pil = self._resize_pil(pil)
            img_b64 = self._pil_to_base64_jpeg(pil)

            try:
                cap = self._caption_one(img_b64)
            except Exception as e:
                cap = f"[caption_error: {e}]"

            caps.append((t, cap))
            evidence.append(FrameEvidence(t_sec=t, tags=["captioned_llava"], extra={}))

        joined = " | ".join([f"t={t:.1f}s: {c}" for t, c in caps])
        return ClipCaption(
            start_sec=timed_frames[0][0],
            end_sec=timed_frames[-1][0],
            caption=joined,
            evidence=evidence,
        )
