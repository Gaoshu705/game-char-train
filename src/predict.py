"""
YOLOv8 推理脚本 - 游戏角色识别
支持: 单张图像推理 / 批量推理 / 视频推理 / 实时摄像头推理
"""
import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import draw_boxes, resolve_path


def load_model(model_path: str) -> YOLO:
    """加载训练好的模型"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    return YOLO(model_path)


def load_class_names(data_yaml: str = "data/dataset.yaml") -> dict:
    """从数据集配置加载类别名"""
    data_path = resolve_path(data_yaml)
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            names = config.get("names", {})
            return {int(k): v for k, v in names.items()}
    return {}


def predict_image(
    model: YOLO,
    image_path: str,
    output_dir: str = "runs/predict",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    show: bool = False,
    save: bool = True,
) -> list:
    """对单张图像进行推理"""
    output_dir = str(resolve_path(output_dir))
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        save=save,
        project=output_dir,
        name="",
        exist_ok=True,
    )
    return results


def predict_batch(
    model: YOLO,
    image_dir: str,
    output_dir: str = "runs/predict_batch",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list:
    """对目录中的图像进行批量推理"""
    output_dir = str(resolve_path(output_dir))
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower() in image_exts
    ]

    if not image_files:
        print(f"[WARNING] 在 {image_dir} 中没有找到图像文件")
        return []

    print(f"找到 {len(image_files)} 张图像，开始推理...")

    results = model.predict(
        source=image_files,
        conf=conf_threshold,
        iou=iou_threshold,
        save=True,
        project=output_dir,
        name="",
        exist_ok=True,
    )

    return results


def predict_video(
    model: YOLO,
    video_path: str,
    output_path: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    show: bool = False,
):
    """对视频进行推理"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height}, {fps:.1f} FPS, {total_frames} 帧")

    writer = None
    if output_path:
        output_path = str(resolve_path(output_path))
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    class_names = load_class_names()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                frame = draw_boxes(frame, boxes, class_ids, scores, class_names, conf_threshold=conf_threshold)

        if writer:
            writer.write(frame)

        if show:
            cv2.imshow("Game Character Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            progress = frame_count / total_frames * 100
            print(f"[{frame_count}/{total_frames}] {progress:.1f}% | {fps_actual:.1f} FPS")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\n处理完成: {frame_count} 帧, 耗时 {elapsed:.1f}s, 平均 {frame_count / elapsed:.1f} FPS")
    if output_path:
        print(f"输出视频: {output_path}")


def list_cameras(max_test: int = 10) -> list:
    """检测并列出可用的摄像头设备

    Args:
        max_test: 最多测试的摄像头 ID 数量

    Returns:
        [(id, name), ...] 可用摄像头列表
    """
    available = []
    print("正在检测可用摄像头...")
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                backend = cap.getBackendName()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                name = f"Camera {i} [{width}x{height}, {backend}]"
                available.append((i, name))
                print(f"  [✓] {name}")
            else:
                print(f"  [✗] Camera {i} - 可打开但无法读取帧")
            cap.release()
        else:
            pass
    if not available:
        print("未检测到可用摄像头！")
    else:
        print(f"\n共检测到 {len(available)} 个可用摄像头")
    return available


def select_camera(auto: bool = False) -> int:
    """选择摄像头：自动选第一个或交互式选择

    Args:
        auto: True 时自动选择第一个摄像头

    Returns:
        摄像头 ID
    """
    cameras = list_cameras()
    if not cameras:
        print("[ERROR] 没有可用摄像头")
        return -1

    if auto or len(cameras) == 1:
        cam_id, cam_name = cameras[0]
        print(f"自动选择: {cam_name}")
        return cam_id

    print("\n请选择摄像头:")
    for idx, (cam_id, cam_name) in enumerate(cameras):
        print(f"  [{idx}] {cam_name}")

    while True:
        try:
            choice = input(f"请输入编号 (0-{len(cameras) - 1}): ").strip()
            idx = int(choice)
            if 0 <= idx < len(cameras):
                cam_id, cam_name = cameras[idx]
                print(f"已选择: {cam_name}")
                return cam_id
            print(f"无效选择，请输入 0-{len(cameras) - 1}")
        except ValueError:
            print("请输入数字")
        except (EOFError, KeyboardInterrupt):
            print("\n已取消")
            return -1


def predict_webcam(
    model: YOLO,
    camera_id: int = 0,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
):
    """实时摄像头推理"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 (ID: {camera_id})")
        return

    class_names = load_class_names()
    print("摄像头已开启，按 'q' 退出...")

    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()

        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                frame = draw_boxes(frame, boxes, class_ids, scores, class_names, conf_threshold=conf_threshold)

        elapsed = time.time() - t_start
        fps_history.append(1.0 / elapsed if elapsed > 0 else 0)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        cv2.putText(
            frame, f"FPS: {avg_fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        cv2.imshow("Game Character Detection - Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="游戏角色识别 - YOLOv8 推理"
    )

    parser.add_argument(
        "--model", type=str, default="runs/train/game_char_train/weights/best.pt",
        help="模型权重路径"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="推理源: 图像路径/目录路径/视频路径/摄像头ID(0,1,2...)"
    )
    parser.add_argument(
        "--camera", type=str, default=None,
        help="摄像头模式: auto=自动选第一个, select=交互选择, 数字=指定摄像头ID"
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="列出所有可用摄像头"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出目录或输出视频路径"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="置信度阈值"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU 阈值 (NMS)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="显示推理结果窗口"
    )
    parser.add_argument(
        "--data", type=str, default="data/dataset.yaml",
        help="数据集配置文件（用于加载类别名）"
    )

    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    model_path = str(resolve_path(args.model))
    model = load_model(model_path)
    print(f"模型已加载: {args.model}")

    source = args.source
    camera_mode = args.camera

    is_camera = camera_mode is not None or (source and source.lower() in ("camera", "webcam"))

    if is_camera or camera_mode:
        if camera_mode and camera_mode.isdigit():
            camera_id = int(camera_mode)
        elif camera_mode == "auto" or (source and source.isdigit()):
            camera_id = int(source) if source and source.isdigit() else 0
        elif camera_mode == "select":
            camera_id = select_camera(auto=False)
            if camera_id < 0:
                return
        elif source and source.isdigit():
            camera_id = int(source)
        elif source and source.lower() in ("camera", "webcam"):
            camera_id = select_camera(auto=False)
            if camera_id < 0:
                return
        else:
            print("[ERROR] 未指定摄像头，使用 --camera auto/select/编号")
            print("        或 --source 0 (指定摄像头ID)")
            return
        out = args.output or f"runs/webcam_{camera_id}"
        predict_webcam(model, camera_id, args.conf, args.iou)
        return

    if not source:
        print("[ERROR] 请指定推理源 --source，或使用 --camera 选择摄像头")
        print("示例:")
        print("  python main.py predict --source image.jpg")
        print("  python main.py predict --source video.mp4")
        print("  python main.py predict --camera auto")
        print("  python main.py predict --camera select")
        print("  python main.py predict --source 0")
        print("  python main.py predict --list-cameras")
        return

    is_video = source.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    is_image = source.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    is_dir = Path(source).is_dir()

    if is_video:
        out = args.output or f"runs/predict_video/{Path(source).stem}.mp4"
        predict_video(model, source, out, args.conf, args.iou, args.show)

    elif is_dir:
        out = args.output or "runs/predict_batch"
        predict_batch(model, source, out, args.conf, args.iou)

    elif is_image:
        out = args.output or "runs/predict"
        predict_image(model, source, out, args.conf, args.iou, args.show)

    else:
        print(f"[ERROR] 不支持的推理源: {source}")
        print("支持的格式: 图像(.jpg/.png), 目录, 视频(.mp4/.avi), 摄像头(--camera)")


if __name__ == "__main__":
    main()
