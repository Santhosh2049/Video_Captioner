# pose_extractor.py
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict

mp_pose = mp.solutions.pose

# list of landmark names we will extract (MediaPipe's naming)
LANDMARKS = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye",
    "right_eye_outer","left_ear","right_ear","mouth_left","mouth_right","left_shoulder",
    "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky",
    "right_pinky","left_index","right_index","left_thumb","right_thumb","left_hip",
    "right_hip","left_knee","right_knee","left_ankle","right_ankle"
]

def extract_keypoints_from_frames(timed_frames: List[Tuple[float, np.ndarray]], model_complexity=1):
    """
    Returns: List[{"t":float, "keypoints": {name: [x,y,z,vis]}}]
    Coordinates are normalized [0..1] relative to image size.
    """
    results = []
    with mp_pose.Pose(static_image_mode=True, model_complexity=model_complexity) as pose:
        for t, frame in timed_frames:
            img_rgb = frame[:, :, ::-1]
            res = pose.process(img_rgb)
            kp_dict = {}
            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks.landmark):
                    if i >= len(LANDMARKS):
                        break
                    kp_dict[LANDMARKS[i]] = [lm.x, lm.y, lm.z, lm.visibility]
            results.append({"t": t, "keypoints": kp_dict})
    return results

# small helper utilities
def keypoint_array(kp_dict, keys=LANDMARKS):
    arr = []
    for k in keys:
        v = kp_dict.get(k)
        if v:
            arr.extend(v[:3])  # x,y,z
        else:
            arr.extend([np.nan, np.nan, np.nan])
    return np.array(arr).reshape(-1, 3)  # (n_landmarks, 3)

def compute_frame_features(kp_dict, image_size=None):
    """
    Basic features: angles (elbows/knees), velocities require prev frame.
    Returns dict of angles (deg), simple distances, and a pose_confidence.
    """
    def angle(a, b, c):
        # angle at b formed by points a-b-c (in degrees)
        a = np.array(a); b = np.array(b); c = np.array(c)
        ba = a - b; bc = c - b
        if np.any(np.isnan(ba)) or np.any(np.isnan(bc)):
            return None
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosang = np.clip(cosang, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    pts = keypoint_array(kp_dict)
    # map names to indices
    name_to_idx = {name: i for i, name in enumerate(LANDMARKS)}
    # example angles
    left_elbow_angle = angle(pts[name_to_idx['left_shoulder']], pts[name_to_idx['left_elbow']], pts[name_to_idx['left_wrist']])
    right_elbow_angle = angle(pts[name_to_idx['right_shoulder']], pts[name_to_idx['right_elbow']], pts[name_to_idx['right_wrist']])
    left_knee_angle = angle(pts[name_to_idx['left_hip']], pts[name_to_idx['left_knee']], pts[name_to_idx['left_ankle']])
    right_knee_angle = angle(pts[name_to_idx['right_hip']], pts[name_to_idx['right_knee']], pts[name_to_idx['right_ankle']])

    # distances normalized by torso length (shoulder to hip)
    def dist(a,b):
        a=np.array(a); b=np.array(b)
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return None
        return float(np.linalg.norm(a-b))

    torso_len = dist(pts[name_to_idx['left_shoulder']], pts[name_to_idx['left_hip']]) or 1.0
    hands_apart = dist(pts[name_to_idx['left_wrist']], pts[name_to_idx['right_wrist']])
    hand_head_left = dist(pts[name_to_idx['left_wrist']], pts[name_to_idx['nose']])

    pose_confidence = np.nanmean([v for _,v in [(k, kp_dict[k][3]) for k in kp_dict]]) if kp_dict else 0.0

    return {
        "angles": {
            "left_elbow": left_elbow_angle,
            "right_elbow": right_elbow_angle,
            "left_knee": left_knee_angle,
            "right_knee": right_knee_angle,
        },
        "distances": {
            "hands_apart": hands_apart / torso_len if hands_apart is not None else None,
            "hand_head_left": hand_head_left / torso_len if hand_head_left is not None else None,
        },
        "pose_confidence": float(pose_confidence)
    }

def compute_temporal_features(timed_kp_list):
    """Compute velocities and deltas across consecutive frames."""
    timed_features = []
    prev_pts = None
    prev_t = None
    for item in timed_kp_list:
        t = item["t"]
        kp = item["keypoints"]
        cur_pts = keypoint_array(kp)
        feats = compute_frame_features(kp)
        if prev_pts is None:
            feats["velocities"] = {}
        else:
            dt = max(1e-3, t - prev_t)
            # wrist velocities as example
            left_wrist_idx = LANDMARKS.index("left_wrist")
            right_wrist_idx = LANDMARKS.index("right_wrist")
            lw = np.linalg.norm(cur_pts[left_wrist_idx] - prev_pts[left_wrist_idx]) / dt
            rw = np.linalg.norm(cur_pts[right_wrist_idx] - prev_pts[right_wrist_idx]) / dt
            feats["velocities"] = {"left_wrist": float(lw), "right_wrist": float(rw)}
        timed_features.append({"t": t, "features": feats})
        prev_pts = cur_pts
        prev_t = t
    return timed_features
