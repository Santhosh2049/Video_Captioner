# run_clip.py
import yaml
from ppe_system.video_io import iter_sampled_frames
from ppe_system.gating import yolo_person_gate, motion_gate
from ppe_system.captioner import BlipCaptioner
from ppe_system.llm_reasoner import understand_video_from_captions
from ppe_system.captioner_florence2 import Florence2Captioner
from ppe_system.captioner_git import GitCaptioner
from ppe_system.captioner_llava_ollama import OllamaLlavaCaptioner
from ppe_system.gating import yolo_person_gate, motion_gate, motion_filter_frames


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a small clip video file")
    ap.add_argument("--config", default="config.yaml")
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

    # 2) Optional gating (useful later when scaling to long videos)
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
            ok = motion_gate(frames)

        if not ok:
            print("Gate: No meaningful human activity detected. Skipping caption/LLM.")
            return
        
    
    # 3) Real captioning (LLaVA via Ollama) on sampled frames
    captioner = OllamaLlavaCaptioner(
        model="minicpm-v",
        base_url="http://localhost:11434",
        prompt=(
            "Describe the frame in 6-10 words. "
        ),
        
        resize_long_side=320,
        jpeg_quality=85,
        timeout_sec=120
    )
    
    # 4) Frame-level motion filtering
    mf_cfg = cfg.get("motion_filter", {})
    if mf_cfg.get("enabled", True):
        filtered_frames = motion_filter_frames(
            timed_frames,
            thresh=mf_cfg.get("thresh", 0.02),
            pre_pad=mf_cfg.get("pre_pad", 1),
            post_pad=mf_cfg.get("post_pad", 1),
        )
        print(f"Motion filter kept {len(filtered_frames)}/{len(timed_frames)} frames.")
    else:
        filtered_frames = timed_frames

    if not filtered_frames:
        print("No motion frames after filtering. Skipping captioning.")
        return
    
    clip_caption = captioner.caption(filtered_frames)
    # 3) Real captioning (BLIP) on sampled frames
    # captioner = BlipCaptioner(
    #     model_name=cfg.get("caption", {}).get("model_name", "Salesforce/blip-image-captioning-base"),
    #     frame_stride=cfg.get("caption", {}).get("frame_stride", 2),
    #     max_frames_to_caption=cfg.get("caption", {}).get("max_frames_to_caption", 1000),
    #     max_new_tokens=cfg.get("caption", {}).get("max_new_tokens", 30),
    # )
    # clip_caption = captioner.caption(timed_frames)

    print("\n=== CLIP CAPTION  ===")
    print(f"{clip_caption.start_sec:.2f}s → {clip_caption.end_sec:.2f}s")
    print(clip_caption.caption)

    # 4) LLM: infer what's happening from the caption timeline
    if cfg.get("llm", {}).get("enabled", True):
        res = understand_video_from_captions(
            clip_caption,
            allowed_actions=cfg["actions"]["allowed"],
            llm_cfg=cfg["llm"],
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
