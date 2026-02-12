from typing import List, Tuple
import numpy as np
from PIL import Image

from .schemas import FrameEvidence, ClipCaption


class GitCaptioner:
    """
    Image captioning using Microsoft GIT.
    Good, stable caption baseline.
    """

    def __init__(
        self,
        model_name: str = "microsoft/git-base-coco",
        device: str | None = None,
        max_new_tokens: int = 40,
        num_beams: int = 3,
    ):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Use fp16 on GPU if available
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def _to_pil(self, frame_bgr: np.ndarray) -> Image.Image:
        rgb = frame_bgr[:, :, ::-1]
        return Image.fromarray(rgb)

    def _caption_image(self, pil_img: Image.Image) -> str:
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )

        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text.strip()

    def caption(self, timed_frames: List[Tuple[float, np.ndarray]]) -> ClipCaption:
        if not timed_frames:
            return ClipCaption(0.0, 0.0, "Empty clip.", [])

        timed_frames = sorted(timed_frames, key=lambda x: x[0])
        evidence = []
        caps = []

        for t, frame in timed_frames:
            cap = self._caption_image(self._to_pil(frame))
            caps.append((t, cap))
            evidence.append(FrameEvidence(t_sec=t, tags=["captioned"], extra={}))

        joined = " | ".join([f"t={t:.1f}s: {c}" for t, c in caps])
        return ClipCaption(
            start_sec=timed_frames[0][0],
            end_sec=timed_frames[-1][0],
            caption=joined,
            evidence=evidence,
        )
