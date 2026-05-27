"""
游戏角色识别 - YOLOv8 项目入口
提供便捷的命令行接口

用法:
    python main.py train              # 训练模型
    python main.py predict            # 推理
    python main.py prepare            # 准备数据集
    python main.py check              # 检查数据集
    python main.py convert            # 转换标注格式
    python main.py env                # 检查环境
    python main.py cleanup            # 清理训练缓存和数据集
    python main.py screen             # 实时屏幕检测（透明覆盖层）
    python main.py extract            # 从视频提取帧
    python main.py gui                # 启动图形界面
"""
import sys
import os
import io

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_train(args):
    from src.train import train
    train(args.config)


def cmd_predict(args):
    from src.predict import main as predict_main
    sys.argv = [
        "predict.py",
        "--model", args.model,
        "--source", args.source,
        "--conf", str(args.conf),
        "--iou", str(args.iou),
    ]
    if args.output:
        sys.argv.extend(["--output", args.output])
    if args.show:
        sys.argv.append("--show")
    predict_main()


def cmd_prepare(args):
    from scripts.prepare_data import main as prepare_main
    sys.argv = [
        "prepare_data.py",
        "--raw_images", args.raw_images,
        "--raw_labels", args.raw_labels,
        "--output", args.output,
        "--seed", str(args.seed),
    ]
    if args.num_classes is not None:
        sys.argv.extend(["--num_classes", str(args.num_classes)])
    prepare_main()


def cmd_check(args):
    from scripts.prepare_data import main as prepare_main
    sys.argv = [
        "prepare_data.py",
        "--output", args.output,
        "--check_only",
    ]
    if args.num_classes is not None:
        sys.argv.extend(["--num_classes", str(args.num_classes)])
    prepare_main()


def cmd_convert(args):
    from scripts.label_converter import main as convert_main
    from src.utils import resolve_path as _resolve
    out_path = str(_resolve(args.output))
    if args.format == "voc2yolo":
        sys.argv = [
            "label_converter.py",
            "voc2yolo",
            "--label_dir", args.label_dir,
            "--output", out_path,
            "--class_map", args.class_map,
        ]
    elif args.format == "coco2yolo":
        sys.argv = [
            "label_converter.py",
            "coco2yolo",
            "--coco_json", args.coco_json,
            "--output", out_path,
        ]
    elif args.format == "help":
        sys.argv = ["label_converter.py", "label_help"]
    convert_main()


def cmd_env(args):
    from src.utils import check_environment
    check_environment()


def cmd_gui(args):
    from src.gui_app import main as gui_main
    gui_main()


def cmd_screen(args):
    from src.screen_detect import ScreenDetector
    import yaml as _yaml

    class_names = {}
    data_yaml = args.data or "data/dataset.yaml"
    if Path(data_yaml).exists():
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f)
            names = cfg.get("names", {})
            class_names = {int(k): v for k, v in names.items()}

    detector = ScreenDetector(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        monitor=args.monitor,
        capture_scale=args.scale,
        target_fps=args.fps,
        class_names=class_names,
        overlay=not args.window,
    )
    detector.run()


def cmd_extract(args):
    from scripts.extract_frames import extract_frames
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        interval=args.interval or 0,
        count=args.count,
        prefix=args.prefix,
        scale=args.scale,
    )


def cmd_cleanup(args):
    import shutil

    base = Path(__file__).resolve().parent

    targets = []

    if args.all:
        targets.append(("runs/", "训练缓存（权重/日志/图表）"))
        targets.append(("data/dataset/images/", "数据集图像划分"))
        targets.append(("data/dataset/labels/", "数据集标注划分"))
        targets.append(("data/processed/", "中间处理数据"))
    else:
        if args.runs:
            targets.append(("runs/", "训练缓存（权重/日志/图表）"))
        if args.dataset:
            targets.append(("data/dataset/images/", "数据集图像划分"))
            targets.append(("data/dataset/labels/", "数据集标注划分"))

    if args.raw:
        targets.append(("data/raw/images/", "原始图像（不可恢复）"))
        targets.append(("data/raw/labels/", "原始标注（不可恢复）"))

    if not targets:
        print("请指定清理目标: --runs / --dataset / --all / --raw")
        print("示例: python main.py cleanup --runs")
        print("      python main.py cleanup --all")
        print("      python main.py cleanup --all --dry-run")
        return

    print("将要清理以下内容:\n")
    for path, desc in targets:
        print(f"  📁 {path}  ({desc})")

    if args.dry_run:
        print("\n[干运行模式] 未执行实际删除。去掉 --dry-run 以执行清理。")
        return

    if args.raw and not args.yes:
        confirm = input("\n⚠️  包含原始数据，删除后不可恢复！输入 yes 确认: ")
        if confirm.lower() != "yes":
            print("已取消。")
            return
    elif not args.yes:
        confirm = input("\n确认清理？[y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("已取消。")
            return

    gitkeep_dirs = [
        "data/dataset/images/train",
        "data/dataset/images/val",
        "data/dataset/images/test",
        "data/dataset/labels/train",
        "data/dataset/labels/val",
        "data/dataset/labels/test",
        "data/raw/images",
        "data/raw/labels",
        "data/processed",
        "models",
        "runs",
    ]

    for path, desc in targets:
        full = base / path
        if full.exists():
            for item in full.iterdir():
                if item.name == ".gitkeep":
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            print(f"  已清理: {path}")

    for d in gitkeep_dirs:
        d_path = base / d
        if d_path.exists() and not list(d_path.iterdir()):
            (d_path / ".gitkeep").touch(exist_ok=True)
        elif not d_path.exists():
            d_path.mkdir(parents=True, exist_ok=True)
            (d_path / ".gitkeep").touch()

    print("\n清理完成。")
    print("如需重新划分数据集: python main.py prepare")


def main():
    parser = argparse.ArgumentParser(
        description="游戏角色识别 - YOLOv8 项目",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--config", default="config/train_config.yaml", help="训练配置文件")

    pred_parser = subparsers.add_parser("predict", help="模型推理")
    pred_parser.add_argument("--model", default="runs/train/game_char_train/weights/best.pt", help="模型路径")
    pred_parser.add_argument("--source", default="0", help="推理源（图像/视频/目录/摄像头ID）")
    pred_parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    pred_parser.add_argument("--iou", type=float, default=0.45, help="IoU阈值")
    pred_parser.add_argument("--output", default=None, help="输出路径")
    pred_parser.add_argument("--show", action="store_true", help="显示结果")

    prepare_parser = subparsers.add_parser("prepare", help="准备数据集")
    prepare_parser.add_argument("--raw_images", default="data/raw/images", help="原始图像目录")
    prepare_parser.add_argument("--raw_labels", default="data/raw/labels", help="原始标注目录")
    prepare_parser.add_argument("--output", default="data/dataset", help="输出目录")
    prepare_parser.add_argument("--num_classes", type=int, default=None,
                                help="类别数（留空从 classes.txt 自动推断）")
    prepare_parser.add_argument("--seed", type=int, default=42, help="随机种子")

    check_parser = subparsers.add_parser("check", help="检查数据集")
    check_parser.add_argument("--output", default="data/dataset", help="数据集目录")
    check_parser.add_argument("--num_classes", type=int, default=None,
                              help="类别数（留空从 classes.txt 推断）")

    convert_parser = subparsers.add_parser("convert", help="标注格式转换")
    convert_parser.add_argument("--format", default="voc2yolo", choices=["voc2yolo", "coco2yolo", "help"])
    convert_parser.add_argument("--label_dir", default=None, help="VOC XML 目录")
    convert_parser.add_argument("--coco_json", default=None, help="COCO JSON 文件")
    convert_parser.add_argument("--output", default="data/raw/labels", help="输出目录")
    convert_parser.add_argument("--class_map", default="{}", help="类别映射")

    env_parser = subparsers.add_parser("env", help="检查运行环境")

    gui_parser = subparsers.add_parser("gui", help="启动图形界面")

    screen_parser = subparsers.add_parser("screen", help="实时屏幕检测（透明覆盖层）")
    screen_parser.add_argument("--model", default="yolov8n.pt", help="模型路径")
    screen_parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    screen_parser.add_argument("--iou", type=float, default=0.45, help="IoU阈值")
    screen_parser.add_argument("--monitor", type=int, default=1, help="显示器编号")
    screen_parser.add_argument("--scale", type=float, default=1.0, help="截图缩放(0.5=加速)")
    screen_parser.add_argument("--fps", type=int, default=30, help="目标帧率")
    screen_parser.add_argument("--data", default="data/dataset.yaml", help="数据集配置")
    screen_parser.add_argument("--window", action="store_true", help="使用窗口模式")

    extract_parser = subparsers.add_parser("extract", help="从视频按时间间隔提取帧")
    extract_parser.add_argument("--video", required=True, help="视频文件路径")
    extract_parser.add_argument("--output", default="data/raw/images", help="输出目录")
    extract_parser.add_argument("--interval", type=float, default=None,
                                help="提取间隔（秒），如 --interval 0.5")
    extract_parser.add_argument("--count", type=int, default=None,
                                help="均匀提取N帧，如 --count 100")
    extract_parser.add_argument("--prefix", default="", help="文件名前缀")
    extract_parser.add_argument("--scale", type=float, default=1.0,
                                help="保存缩放比例")

    cleanup_parser = subparsers.add_parser("cleanup", help="清理训练缓存和数据集")
    cleanup_parser.add_argument("--runs", action="store_true", help="清理训练缓存")
    cleanup_parser.add_argument("--dataset", action="store_true", help="清理数据集划分")
    cleanup_parser.add_argument("--all", action="store_true", help="清理训练缓存 + 数据集划分")
    cleanup_parser.add_argument("--raw", action="store_true", help="同时清理原始数据（需配合 --all 或 --dataset）")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际删除")
    cleanup_parser.add_argument("--yes", "-y", action="store_true", help="跳过确认提示")

    args = parser.parse_args()

    commands = {
        "train": cmd_train,
        "predict": cmd_predict,
        "prepare": cmd_prepare,
        "check": cmd_check,
        "convert": cmd_convert,
        "env": cmd_env,
        "gui": cmd_gui,
        "screen": cmd_screen,
        "extract": cmd_extract,
        "cleanup": cmd_cleanup,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
