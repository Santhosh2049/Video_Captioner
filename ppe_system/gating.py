import numpy as np
from typing import List, Tuple

def motion_gate(frames: List[np.ndarray], thresh: float = 0.02) -> bool:
    # very simple motion energy based on frame difference
    if len(frames) < 2:
        return False
    diffs = []
    prev = frames[0].astype(np.float32)
    for f in frames[1:]:
        cur = f.astype(np.float32)
        diff = np.mean(np.abs(cur - prev)) / 255.0
        diffs.append(diff)
        prev = cur
    return (np.mean(diffs) if diffs else 0.0) >= thresh

def yolo_person_gate(timed_frames: List[Tuple[float, np.ndarray]], model_name: str, person_conf: float, min_person_frames: int) -> bool:
    from ultralytics import YOLO

    model = YOLO(model_name)
    person_count = 0

    for _, frame in timed_frames:
        res = model.predict(frame, verbose=False)[0]
        if res.boxes is None:
            continue

        # COCO person class id = 0 in YOLO models
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        person_present = np.any((cls == 0) & (conf >= person_conf))
        if person_present:
            person_count += 1

    return person_count >= min_person_frames

import numpy as np
from typing import List, Tuple

def motion_filter_frames(
    timed_frames: List[Tuple[float, np.ndarray]],
    thresh: float = 0.02,      # same scale as motion_gate (normalized 0..1)
    pre_pad: int = 1,          # keep 1 frame before motion
    post_pad: int = 1,         # keep 1 frame after motion
) -> List[Tuple[float, np.ndarray]]:
    """
    Frame-level motion filter.
    Keeps frames whose diff from previous frame exceeds thresh,
    with optional padding for context.

    Uses the SAME diff metric as motion_gate():
      mean(abs(cur-prev))/255.0
    """
    if len(timed_frames) < 2:
        return timed_frames

    keep = [False] * len(timed_frames)

    prev = timed_frames[0][1].astype(np.float32)
    for i in range(1, len(timed_frames)):
        cur = timed_frames[i][1].astype(np.float32)
        diff = float(np.mean(np.abs(cur - prev)) / 255.0)
        if diff >= thresh:
            keep[i] = True
        prev = cur

    # add context padding
    for i, k in enumerate(keep):
        if k:
            for j in range(max(0, i - pre_pad), min(len(keep), i + post_pad + 1)):
                keep[j] = True

    filtered = [timed_frames[i] for i in range(len(timed_frames)) if keep[i]]
    return filtered
