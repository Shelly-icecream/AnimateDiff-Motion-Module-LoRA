import cv2
import numpy as np


def downsample_gray(frame_bgr, scale=0.5):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        h, w = gray.shape
        gray = cv2.resize(gray, (max(1, int(w*scale)), max(1, int(h*scale))))
    return gray

def compute_slowmo_score(video_path, fps_sample=30, min_frames=12, scale=0.5):
    """
    Returns: (score_norm, fps)
    score_norm: A score between 0 and 1; higher values indicate slower motion.
    """

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1:
        cap.release()
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < min_frames:
        cap.release()
        return None

    step = max(1, int(round(fps / fps_sample)))

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return None

    prev_gray = downsample_gray(prev, scale=scale)

    mags = []
    accel = []

    last_mag = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cur_idx % step != 0:
            continue

        gray = downsample_gray(frame, scale=scale)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        m = float(np.mean(mag))
        mags.append(m)

        if last_mag is not None:
            accel.append(abs(m - last_mag))
        last_mag = m

        prev_gray = gray

    cap.release()

    if len(mags) < 5:
        return None

    mags = np.array(mags, dtype=np.float32)
    accel = np.array(accel, dtype=np.float32) if len(accel) > 0 else np.array([0.0])

    mean_flow = float(np.mean(mags))
    mean_accel = float(np.mean(accel))

    flow_norm = mean_flow / (mean_flow + 1.0)
    accel_norm = mean_accel / (mean_accel + 0.5)

    S1 = 1.0 - flow_norm
    S2 = 1.0 - accel_norm

    if fps >= 120:
        S3 = 1.0
    elif fps >= 60:
        S3 = 0.5
    else:
        S3 = 0.0

    score = 0.6 * S1 + 0.3 * S2 + 0.1 * S3
    score = float(np.clip(score, 0.0, 1.0))

    return score, fps
