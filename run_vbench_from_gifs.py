import os
import re
import csv
import json
import shlex
import shutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

# 1) GIF 输入目录（可以包含子目录）
GIF_ROOT = r"D:\samples\coastline"

# 2) 实验输出目录（脚本会自动创建）
EXP_ROOT = r"D:\AnimateDiff\vbench_experiments\exp_001"

# 3) 转码规格（科研可复现）
FPS = 8                 # VBench 常用 8fps
SHORT_EDGE = 512        # 统一短边 512（常用）
CRF = 18                # 质量：18~23；越小越清晰也越大
PRESET = "medium"       # 编码速度/质量权衡：slow更大更慢

# 4) VBench 运行命令（按你本地实际情况改）
# 例子1：在 vbench repo 根目录运行 eval.py
VBENCH_WORKDIR = r"D:\VBench-master"
VBENCH_CMD = (
    r"python eval.py "
    r'--videos_path "{videos_dir}" '
    r'--output_path "{results_dir}"'
)

# 如果你的 VBench 命令不同，比如叫 run.py 或需要额外参数，
# 就把上面 VBENCH_CMD 改成你能在命令行跑通的那条。



def run(cmd, cwd=None):
    print(f"[CMD] {cmd}")
    p = subprocess.run(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{p.stdout}")
    return p.stdout

def which(exe):
    return shutil.which(exe) is not None

def sha1_file(path: Path):
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_name(s: str):
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
    return s[:120] if len(s) > 120 else s

def ffprobe_info(mp4: Path):
    # 取关键字段：codec_name, pix_fmt, r_frame_rate, width/height
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt,r_frame_rate,width,height -of json "{mp4}"'
    out = subprocess.check_output(cmd, shell=True, text=True)
    j = json.loads(out)
    st = j["streams"][0]
    return st

def convert_gif_to_mp4(gif: Path, mp4: Path):
    vf = f"fps={FPS},scale='if(lt(iw,ih),{SHORT_EDGE},-2)':'if(lt(iw,ih),-2,{SHORT_EDGE})':flags=lanczos"
    cmd = (
        f'ffmpeg -y -hide_banner -loglevel error -i "{gif}" '
        f'-vf "{vf}" -c:v libx264 -preset {PRESET} -crf {CRF} '
        f'-pix_fmt yuv420p -movflags +faststart "{mp4}"'
    )
    run(cmd)

def discover_gifs(root: Path):
    return sorted([p for p in root.rglob("*.gif")])

def main():
    if not which("ffmpeg"):
        raise SystemExit("找不到 ffmpeg：请先把 ffmpeg/bin 加到 PATH，然后重新打开终端。")
    if not which("ffprobe"):
        raise SystemExit("找不到 ffprobe：ffmpeg 安装包一般自带 ffprobe，请确认 PATH 配置正确。")

    gif_root = Path(GIF_ROOT)
    exp_root = Path(EXP_ROOT)
    exp_root.mkdir(parents=True, exist_ok=True)

    # 输出结构
    videos_dir = exp_root / "videos"
    results_dir = exp_root / "vbench_results"
    logs_dir = exp_root / "logs"
    videos_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # 记录本次实验配置（科研复现）
    meta = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "gif_root": str(gif_root),
        "exp_root": str(exp_root),
        "fps": FPS,
        "short_edge": SHORT_EDGE,
        "crf": CRF,
        "preset": PRESET,
        "vbench_workdir": VBENCH_WORKDIR,
        "vbench_cmd_template": VBENCH_CMD,
    }
    (exp_root / "experiment_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    gifs = discover_gifs(gif_root)
    if not gifs:
        raise SystemExit(f"在 {gif_root} 下没找到 .gif")

    # 转换清单 CSV（可审计）
    manifest_csv = exp_root / "manifest.csv"
    rows = []
    print(f"[INFO] Found {len(gifs)} GIFs")


    # 目录规则：videos/<group>/<id>.mp4
    # group = 相对路径的父目录名（没有就 root）
    for idx, gif in enumerate(gifs, start=1):
        rel = gif.relative_to(gif_root)
        group = safe_name(rel.parent.as_posix()) if str(rel.parent) != "." else "root"
        sample_id = f"{idx:04d}"
        out_dir = videos_dir / group
        out_dir.mkdir(parents=True, exist_ok=True)
        mp4 = out_dir / f"{sample_id}.mp4"

        print(f"[{idx}/{len(gifs)}] {gif} -> {mp4}")
        convert_gif_to_mp4(gif, mp4)

        # 质量/兼容性检查
        info = ffprobe_info(mp4)
        ok = (info.get("codec_name") == "h264" and info.get("pix_fmt") == "yuv420p")
        if not ok:
            raise RuntimeError(f"转码后格式不符合预期：{mp4}\n{info}")

        rows.append({
            "index": idx,
            "gif_path": str(gif),
            "gif_sha1": sha1_file(gif),
            "group": group,
            "mp4_path": str(mp4),
            "mp4_sha1": sha1_file(mp4),
            "width": info.get("width"),
            "height": info.get("height"),
            "r_frame_rate": info.get("r_frame_rate"),
            "codec": info.get("codec_name"),
            "pix_fmt": info.get("pix_fmt"),
        })

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 运行 VBench
    cmd = VBENCH_CMD.format(videos_dir=str(videos_dir), results_dir=str(results_dir))
    log_path = logs_dir / "vbench_run.log"
    print("[INFO] Running VBench...")
    out = run(cmd, cwd=VBENCH_WORKDIR)
    log_path.write_text(out, encoding="utf-8")

    print("\n[DONE] 全流程完成：")
    print(f" - Videos:  {videos_dir}")
    print(f" - Results: {results_dir}")
    print(f" - Manifest:{manifest_csv}")
    print(f" - Log:     {log_path}")

if __name__ == "__main__":
    main()