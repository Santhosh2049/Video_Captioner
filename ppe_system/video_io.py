import cv2
import numpy as np
from typing import List, Tuple

def iter_sampled_frames(video_path: str, sample_fps: float, resize_width: int = 640, max_frames: int = 0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / sample_fps)), 1)

    idx = 0
    yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step == 0:
            h, w = frame.shape[:2]
            if resize_width and w > resize_width:
                new_h = int(h * (resize_width / w))
                frame = cv2.resize(frame, (resize_width, new_h))

            t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield t_sec, frame
            yielded += 1
            if max_frames and yielded >= max_frames:
                break

        idx += 1

    cap.release()
