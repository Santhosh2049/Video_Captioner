from dataclasses import dataclass
from typing import List, Dict

@dataclass
class FrameEvidence:
    t_sec: float
    tags: List[str]
    extra: Dict[str, float]

@dataclass
class ClipCaption:
    start_sec: float
    end_sec: float
    caption: str              # timeline text: "t=0.0s: ... | t=2.0s: ..."
    evidence: List[FrameEvidence]

@dataclass
class VideoUnderstanding:
    action: str               # from allowed_actions
    confidence: float
    summary: str              # what is happening overall
    key_events: List[Dict]    # [{"time":"t=2.4s","event":"..."}]
