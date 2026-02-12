import yaml
import numpy as np
from ppe_system.video_io import iter_sampled_frames, sample_frames_in_windows
from ppe_system.gating import yolo_person_gate, motion_gate
from ppe_system.captioner import BlipCaptioner
from ppe_system.llm_reasoner import understand_video_from_captions


def compute_motion_scores(timed_frames):
    """
    Returns list of (t_sec, motion_score) between consecutive sampled frames.
    motion_score ~ mean absolute pixel diff normalized [0..1]
    """
    if len(timed_frames) < 2:
        return []

    scores = []
    prev_t, prev_f = timed_frames[0]
    prev = prev_f.astype(np.float32)

    for t, frame in timed_frames[1:]:
        cur = frame.astype(np.float32)
        m = float(np.mean(np.abs(cur - prev)) / 255.0)
        scores.append((t, m))
        prev = cur
        prev_t = t

    return scores


def build_burst_windows(base_times, motion_scores, motion_threshold, window_sec, max_windows):
    """
    Creates time windows around high-motion moments.
    """
    if not motion_scores:
        return []

    half = window_sec / 2.0
    windows = []

    for t, m in motion_scores:
        if m >= motion_threshold:
            windows.append((max(0.0, t - half), t + half))
            if len(windows) >= max_windows:
                break

    # Merge overlapping windows
    if not windows:
        return []
    windows.sort()
    merged = [windows[0]]
    for a, b in windows[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))

    return merged


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a video file")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    # 1) Base sampling at 1 FPS (or cfg sample_fps)
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

    timed_frames.sort(key=lambda x: x[0])

    # 4) Motion-triggered burst sampling (extra frames only where motion is high)
    burst_cfg = cfg.get("burst", {})
    if burst_cfg.get("enabled", False):
        motion_scores = compute_motion_scores(timed_frames)

        windows = build_burst_windows(
            base_times=[t for t, _ in timed_frames],
            motion_scores=motion_scores,
            motion_threshold=burst_cfg.get("motion_threshold", 0.035),
            window_sec=burst_cfg.get("window_sec", 1.0),
            max_windows=burst_cfg.get("max_burst_windows", 10),
        )

        if windows:
            extra = sample_frames_in_windows(
                args.video,
                windows=windows,
                burst_fps=burst_cfg.get("burst_fps", 3),
                resize_width=cfg["video"]["resize_width"],
                max_total_frames=5000
            )

            # Combine + dedupe by timestamp (keep the first frame for a timestamp)
            combined = {}
            for t, f in timed_frames:
                combined[round(t, 3)] = (t, f)
            for t, f in extra:
                key = round(t, 3)
                if key not in combined:
                    combined[key] = (t, f)

            timed_frames = sorted(combined.values(), key=lambda x: x[0])
            print(f"\n[BURST] Added {max(0, len(timed_frames) - len(motion_scores) - 1)} extra frames in windows: {windows}")

    # 3) Caption all frames we decided to keep (1 fps + bursts)
    caption_cfg = cfg.get("caption", {})
    captioner = BlipCaptioner(
        model_name=caption_cfg.get("model_name", "Salesforce/blip-image-captioning-base"),
        max_new_tokens=caption_cfg.get("max_new_tokens", 30),
    )

    clip_caption = captioner.caption(
        timed_frames,
        merge_similar=caption_cfg.get("merge_similar", True),
        merge_threshold=caption_cfg.get("merge_threshold", 0.92),
    )

    print("\n=== CLIP CAPTION (BLIP) ===")
    print(f"{clip_caption.start_sec:.2f}s → {clip_caption.end_sec:.2f}s")
    print(clip_caption.caption)

    # LLM: infer what's happening from the caption timeline
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
    else:
        print("\nLLM disabled in config.yaml (llm.enabled: false).")


if __name__ == "__main__":
    main()
