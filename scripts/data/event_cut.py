import cv2
import numpy as np
from tqdm import tqdm
import os
import glob

# ========= 工具函数 =========
def get_videos_glob(folder_path):
    video_files = []
    patterns = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    for pattern in patterns:
        video_files.extend(glob.glob(os.path.join(folder_path, pattern)))
    return video_files

def moving_average(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="same")

def iou_1d(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(1e-9, (a1 - a0) + (b1 - b0) - inter)
    return inter / union

# ========= 参数区 =========
INPUT_DIR = r"D:\AnimateDiff\raw"
OUTPUT_DIR = r"D:\AnimateDiff\clips"

CLIP_SECONDS = 5
FPS_SAMPLE = 30
MAX_CLIPS = 5

THRESH_PERCENTILE = 92            # 用分位数做阈值，越大越严格(90~97建议)
SMOOTH_WINDOW = 5                 # 平滑窗口（采样点）
MIN_EVENT_SECONDS = 0.4           # 事件最短持续时间(秒)
MIN_GAP_SECONDS = 1.0             # 事件之间最小间隔(秒)，太近就合并
MAX_EVENT_SECONDS = 3.0           # 事件最长持续时间(秒)，太长就只取最强子段
CLIP_IOU_DEDUP = 0.55             # clip重叠超过这个比例就认为重复

os.makedirs(OUTPUT_DIR, exist_ok=True)

videos = get_videos_glob(INPUT_DIR)
print(f"🎬 发现 {len(videos)} 个视频")

# ========= 主循环 =========
for video_path in videos:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n▶ 处理视频: {video_name}")

    video_out_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(video_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        print("⚠️ 无法读取视频，跳过")
        cap.release()
        continue

    sample_step = max(1, int(round(fps / FPS_SAMPLE)))

    ret, prev = cap.read()
    if not ret:
        cap.release()
        continue

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flow_energy = []
    sample_frame_ids = []

    print("  计算光流能量中...")

    for i in tqdm(range(1, frame_count), leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        if i % sample_step != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_energy.append(float(mag.mean()))
        sample_frame_ids.append(i)
        prev_gray = gray

    cap.release()

    flow_energy = np.array(flow_energy, dtype=np.float32)

    if len(flow_energy) < 10:
        print("❌ 有效采样帧太少，跳过")
        continue

    # ========= 平滑 + 阈值 =========
    flow_smooth = moving_average(flow_energy, SMOOTH_WINDOW)

    # 分位数阈值，比 max*ratio 稳定太多
    threshold = np.percentile(flow_smooth, THRESH_PERCENTILE)

    event_indices = np.where(flow_smooth >= threshold)[0]
    if len(event_indices) == 0:
        print("❌ 未检测到明显事件")
        continue

    # ========= 合并事件（按时间间隔） =========
    # 把 index -> time
    sample_times = np.array(sample_frame_ids) / fps
    idx_to_time = lambda idx: float(sample_times[idx])

    min_gap = MIN_GAP_SECONDS
    events = []
    current = [event_indices[0]]

    for idx in event_indices[1:]:
        prev_idx = current[-1]
        if idx_to_time(idx) - idx_to_time(prev_idx) <= (sample_step / fps + min_gap):
            current.append(idx)
        else:
            events.append(current)
            current = [idx]
    events.append(current)

    # ========= 过滤事件长度 =========
    min_event = MIN_EVENT_SECONDS
    filtered = []
    for ev in events:
        t0 = idx_to_time(ev[0])
        t1 = idx_to_time(ev[-1])
        if (t1 - t0) >= min_event:
            filtered.append(ev)

    if len(filtered) == 0:
        print("❌ 事件都太短（噪声），跳过")
        continue

    # ========= 给事件打分（强度）并排序 =========
    scored = []
    for ev in filtered:
        # event score 用 smooth 的均值/最大值都行，这里用 mean 更稳
        score = float(flow_smooth[ev].mean())
        t0 = idx_to_time(ev[0])
        t1 = idx_to_time(ev[-1])
        scored.append((score, ev, t0, t1))

    scored.sort(key=lambda x: x[0], reverse=True)

    # ========= 切视频 =========
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    clip_id = 0
    used_windows = []  # [(start, end)]

    for score, ev, t0, t1 in scored:
        if clip_id >= MAX_CLIPS:
            break

        # 如果事件太长，只取事件中心附近（防止切出“全程都在动”的无聊段）
        center_time = float((t0 + t1) / 2.0)
        start_time = max(0.0, center_time - CLIP_SECONDS / 2)
        end_time = start_time + CLIP_SECONDS

        # ========= 去重：和已有 clip 重叠太多就跳过 =========
        duplicate = False
        for (u0, u1) in used_windows:
            if iou_1d(start_time, end_time, u0, u1) >= CLIP_IOU_DEDUP:
                duplicate = True
                break
        if duplicate:
            continue

        # ========= 真正写入 =========
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        out_path = os.path.join(video_out_dir, f"clip_{clip_id:02d}.mp4")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frames_to_write = int(round(CLIP_SECONDS * fps))
        for _ in range(frames_to_write):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        used_windows.append((start_time, end_time))
        clip_id += 1

    cap.release()
    print(f"✅ 生成 {clip_id} 个 clips")
