from typing import List, Tuple
import numpy as np
from PIL import Image

from .schemas import FrameEvidence, ClipCaption


class Florence2Captioner:
    """
    Florence-2 captioner using official Transformers support (no trust_remote_code).
    Uses task prompts like <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>.  :contentReference[oaicite:2]{index=2}
    """

    def __init__(
        self,
        model_name: str = "florence-community/Florence-2-base",
        task_prompt: str = "<DETAILED_CAPTION>",
        max_new_tokens: int = 64,
        num_beams: int = 3,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.task_prompt = task_prompt
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def _to_pil(self, frame_bgr: np.ndarray) -> Image.Image:
        rgb = frame_bgr[:, :, ::-1]
        return Image.fromarray(rgb)

    def _caption_image(self, pil_img: Image.Image) -> str:
        # Florence-2 is prompt-based: <CAPTION>, <DETAILED_CAPTION>, ...  :contentReference[oaicite:3]{index=3}
        task_prompt = self.task_prompt
        image_size = pil_img.size

        inputs = self.processor(text=task_prompt, images=pil_img, return_tensors="pt")

        # Move tensors to device
        for k in list(inputs.keys()):
            inputs[k] = inputs[k].to(self.device)

        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Try official post-processing (returns structured dict)
        try:
            parsed = self.processor.post_process_generation(
                generated_text, task=task_prompt, image_size=image_size
            )
            # parsed usually contains the caption under a key matching the task
            if isinstance(parsed, dict):
                # handle common shapes
                val = parsed.get(task_prompt, None)
                if isinstance(val, str):
                    return val.strip()
                if isinstance(val, list) and val and isinstance(val[0], str):
                    return val[0].strip()
            # fallback
        except Exception:
            pass

        # Fallback: return raw generated text stripped
        return generated_text.strip()

    def caption(self, timed_frames: List[Tuple[float, np.ndarray]]) -> ClipCaption:
        if not timed_frames:
            return ClipCaption(0.0, 0.0, "Empty clip.", [])

        timed_frames = sorted(timed_frames, key=lambda x: x[0])
        evidence = []
        captions = []

        for t, frame in timed_frames:
            cap = self._caption_image(self._to_pil(frame))
            captions.append((t, cap))
            evidence.append(FrameEvidence(t_sec=t, tags=["captioned"], extra={}))

        joined = " | ".join([f"t={t:.1f}s: {c}" for t, c in captions])
        return ClipCaption(
            start_sec=timed_frames[0][0],
            end_sec=timed_frames[-1][0],
            caption=joined,
            evidence=evidence,
        )
