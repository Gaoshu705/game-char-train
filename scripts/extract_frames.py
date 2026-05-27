"""
视频帧提取脚本
按指定时间间隔从视频中提取帧，保存为图片

用法:
  python scripts/extract_frames.py --video game.mp4 --interval 1
  python scripts/extract_frames.py --video game.mp4 --interval 0.5 --output data/raw/images
  python scripts/extract_frames.py --video game.mp4 --count 50    # 均匀提取50帧
"""

import argparse
import math
from pathlib import Path

import cv2
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import resolve_path


def extract_frames(
    video_path: str,
    output_dir: str = "data/raw/images",
    interval: float = 1.0,
    count: int = None,
    prefix: str = "",
    scale: float = 1.0,
):
    """
    从视频中提取帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        interval: 提取间隔（秒），与 count 二选一
        count: 提取总帧数（均匀采样），与 interval 二选一
        prefix: 输出文件名前缀
        scale: 保存时的缩放比例
    """
    video = Path(video_path)
    if not video.exists():
        print(f"[ERROR] 视频文件不存在: {video_path}")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"\n视频信息:")
    print(f"  文件: {video.name}")
    print(f"  分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.2f}s\n")

    if prefix:
        prefix = prefix + "_"

    if count is not None and count > 0:
        if count >= total_frames:
            print(f"[WARNING] 请求帧数 {count} 超过总帧数 {total_frames}，将提取所有帧")
            frame_indices = list(range(total_frames))
        else:
            step = (total_frames - 1) / (count - 1) if count > 1 else 0
            frame_indices = [int(i * step) for i in range(count)]

        mode = f"均匀 {count} 帧"
    else:
        if interval <= 0:
            print("[ERROR] 间隔必须大于 0")
            cap.release()
            return

        frame_step = int(fps * interval)
        if frame_step <= 0:
            frame_step = 1

        frame_indices = list(range(0, total_frames, frame_step))
        mode = f"每 {interval}s"

    print(f"提取模式: {mode}")
    print(f"将提取 {len(frame_indices)} 帧到: {output_dir}\n")

    saved = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp = frame_idx / fps if fps > 0 else frame_idx

        if scale != 1.0:
            new_w = int(frame.shape[1] * scale)
            new_h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        filename = f"{prefix}{video.stem}_frame_{frame_idx:06d}_{timestamp:.2f}s.jpg"
        out_path = out_dir / filename
        cv2.imwrite(str(out_path), frame)
        saved += 1

        if saved % 50 == 0:
            progress = saved / len(frame_indices) * 100
            print(f"进度: {saved}/{len(frame_indices)} ({progress:.1f}%)")

    cap.release()
    print(f"\n提取完成: 共 {saved} 帧 → {output_dir}")
    print(f"接下来可以用 labelImg 打开该目录进行标注。")


def main():
    parser = argparse.ArgumentParser(
        description="从视频中按时间间隔提取帧"
    )
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--output", default="data/raw/images", help="输出目录")
    parser.add_argument("--interval", type=float, default=None,
                        help="提取间隔（秒），如 --interval 0.5 每0.5秒1帧")
    parser.add_argument("--count", type=int, default=None,
                        help="均匀提取N帧，如 --count 100")
    parser.add_argument("--prefix", default="", help="文件名前缀")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="保存缩放比例 (0.5=半分辨率)")
    args = parser.parse_args()

    if args.interval is None and args.count is None:
        args.interval = 1.0

    extract_frames(
        video_path=args.video,
        output_dir=str(resolve_path(args.output)),
        interval=args.interval or 0,
        count=args.count,
        prefix=args.prefix,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
