import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from animatediff.data.slowmo import compute_slowmo_score


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir,
        n_frames=8,
        size=256,
        stride=1,
        debug_stride_print=False,

        enable_slowmo_sampling=True,
        slowmo_fps_sample=30,
        slowmo_min_score=0.35,   
        slowmo_cache=True
    ):
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.size = size
        self.stride = stride
        self.debug_stride_print = debug_stride_print

        self.enable_slowmo_sampling = enable_slowmo_sampling
        self.slowmo_fps_sample = slowmo_fps_sample
        self.slowmo_min_score = slowmo_min_score
        self.slowmo_cache = slowmo_cache

        self.video_paths = glob.glob(os.path.join(root_dir, "**", "*.mp4"), recursive=True)
        print(f"Found {len(self.video_paths)} videos")
        print(f"Dataset config: n_frames={self.n_frames}, size={self.size}, stride={self.stride}")

        self.slowmo_scores = None
        self.valid_paths = self.video_paths

        if self.enable_slowmo_sampling:
            self._build_slowmo_cache()

    def _build_slowmo_cache(self):
        print("Computing slowmo scores (one-time)...")

        scores = []
        paths = []

        self.slowmo_tags = {}

        for p in tqdm(self.video_paths):
            out = compute_slowmo_score(
                p,
                fps_sample=self.slowmo_fps_sample
            )

            if out is None:
                continue

            score, fps = out

            if score < self.slowmo_min_score:
                continue

            paths.append(p)
            scores.append(score)

            self.slowmo_tags[p] = "<slowmo>"

        if len(paths) == 0:
            print("No clips passed slowmo filter. Fallback to full dataset.")
            self.valid_paths = self.video_paths
            self.slowmo_scores = None
            self.slowmo_tags = {}
            return

        self.valid_paths = paths
        self.slowmo_scores = np.array(scores, dtype=np.float32)

        print(f"Slowmo filter kept {len(self.valid_paths)} / {len(self.video_paths)} clips")
        print(f"slowmo score stats: "
              f"min={self.slowmo_scores.min():.3f}, "
              f"mean={self.slowmo_scores.mean():.3f}, "
              f"max={self.slowmo_scores.max():.3f}")

    def __len__(self):
        return len(self.valid_paths)

    def _sample_index(self):
        if (not self.enable_slowmo_sampling) or (self.slowmo_scores is None):
            return random.randint(0, len(self.valid_paths) - 1)

        temp = 3.0
        w = np.power(self.slowmo_scores + 1e-3, temp)
        w = w / w.sum()
        return int(np.random.choice(len(self.valid_paths), p=w))

    def __getitem__(self, idx):
        for _ in range(10):
            real_idx = self._sample_index()
            path = self.valid_paths[real_idx]

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            need = (self.n_frames - 1) * self.stride + 1
            if total < need or total <= 0:
                cap.release()
                continue

            start = random.randint(0, total - need)

            frames = []
            frame_indices = []

            ok = True
            for i in range(self.n_frames):
                frame_idx = start + i * self.stride
                frame_indices.append(frame_idx)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    ok = False
                    break

                frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
                frame = frame[:, :, ::-1]  # BGR -> RGB
                frames.append(frame)

            cap.release()

            if not ok or len(frames) != self.n_frames:
                continue

            if self.debug_stride_print and random.random() < 0.02:
                print(f"STRIDE CHECK: stride={self.stride}, total={total}, start={start}")
                print(f"selected frame idx = {frame_indices}")

            frames = np.stack(frames, axis=0)  # [F,H,W,3]
            frames = torch.from_numpy(frames).float() / 255.0
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [F,3,H,W]

            txt_path = path.replace(".mp4", ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    base_prompt = f.read().strip()
            else:
                base_prompt = ""

            # Add the slow-motion tag if sampling is enabled and the video passed the filter.
            if self.enable_slowmo_sampling and path in self.slowmo_tags:
                prompt = "<slowmo> " + base_prompt
            else:
                prompt = base_prompt

            return {"pixel_values": frames, "text": prompt}

        raise RuntimeError("Failed to fetch a valid video sample after 10 retries.")
