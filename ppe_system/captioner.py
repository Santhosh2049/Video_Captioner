from typing import List, Tuple
import numpy as np
from PIL import Image

from .schemas import FrameEvidence, ClipCaption


class BlipCaptioner:
    """
    Real image captioner using BLIP.
    Strategy:
      - sample a few frames from the clip
      - caption each sampled frame
      - join captions into a compact clip caption
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str | None = None,
        max_new_tokens: int = 30,
        frame_stride: int = 4,   # caption every Nth sampled frame
        max_frames_to_caption: int = 8
    ):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch

        self.torch = torch
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames_to_caption = max(1, int(max_frames_to_caption))

    def _to_pil(self, frame_bgr: np.ndarray) -> Image.Image:
        # OpenCV frame is BGR; PIL expects RGB
        rgb = frame_bgr[:, :, ::-1]
        return Image.fromarray(rgb)

    def _caption_image(self, pil_img: Image.Image) -> str:
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return text.strip()

    def caption(self, timed_frames: List[Tuple[float, np.ndarray]]) -> ClipCaption:
        if not timed_frames:
            return ClipCaption(0.0, 0.0, "Empty clip.", [])

        times = [t for t, _ in timed_frames]

        # Pick frames to caption: every stride, cap total
        picked = []
        for i, (t, frame) in enumerate(timed_frames):
            if i % self.frame_stride == 0:
                picked.append((t, frame))
            if len(picked) >= self.max_frames_to_caption:
                break

        evidence: List[FrameEvidence] = []
        captions = []

        for t, frame in picked:
            pil_img = self._to_pil(frame)
            cap = self._caption_image(pil_img)
            captions.append((t, cap))
            evidence.append(FrameEvidence(t_sec=t, tags=["captioned"], extra={}))

        # Build a clip-level caption (compact)
        # Example: "t=0.5s: ... | t=2.5s: ... | ..."
        joined = " | ".join([f"t={t:.1f}s: {c}" for t, c in captions])

        return ClipCaption(
            start_sec=min(times),
            end_sec=max(times),
            caption=joined,
            evidence=evidence
        )
