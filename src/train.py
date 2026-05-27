"""
YOLOv8 训练脚本 - 游戏角色识别
"""
import os
import sys
import argparse
import warnings
from pathlib import Path

import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, setup_logging, check_environment, resolve_path


def train(config_path: str = "config/train_config.yaml"):
    """
    执行 YOLOv8 训练

    Args:
        config_path: 训练配置文件路径（相对路径基于项目根目录）
    """
    config_path = resolve_path(config_path)
    config = load_config(config_path)

    project_dir = str(resolve_path(config.get("project", "runs/train")))
    logger = setup_logging(
        str(resolve_path(config.get("project", "runs/train"), "logs")),
        config.get("name", "game_char_train"),
    )

    logger.info("=" * 60)
    logger.info("游戏角色识别 - YOLOv8 训练")
    logger.info("=" * 60)

    data_yaml = str(resolve_path(config.get("data", "data/dataset.yaml")))

    if not Path(data_yaml).exists():
        logger.error(f"数据集配置文件不存在: {data_yaml}")
        logger.info("请先创建数据集并运行 scripts/prepare_data.py")
        return

    with open(data_yaml, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    num_classes = data_config.get("nc", 7)
    class_names = data_config.get("names", {})
    logger.info(f"类别数量: {num_classes}")
    logger.info(f"类别列表: {class_names}")

    model_name = config.get("model", "yolov8n.pt")
    logger.info(f"加载模型: {model_name}")

    model = YOLO(model_name)

    train_args = {
        "data": data_yaml,
        "epochs": config.get("epochs", 100),
        "batch": config.get("batch", 16),
        "imgsz": config.get("imgsz", 640),
        "patience": config.get("patience", 20),
        "save": config.get("save", True),
        "save_period": config.get("save_period", 10),
        "optimizer": config.get("optimizer", "AdamW"),
        "lr0": config.get("lr0", 0.001),
        "lrf": config.get("lrf", 0.01),
        "momentum": config.get("momentum", 0.937),
        "weight_decay": config.get("weight_decay", 0.0005),
        "hsv_h": config.get("hsv_h", 0.015),
        "hsv_s": config.get("hsv_s", 0.7),
        "hsv_v": config.get("hsv_v", 0.4),
        "degrees": config.get("degrees", 0.0),
        "translate": config.get("translate", 0.1),
        "scale": config.get("scale", 0.5),
        "shear": config.get("shear", 0.0),
        "perspective": config.get("perspective", 0.0),
        "flipud": config.get("flipud", 0.0),
        "fliplr": config.get("fliplr", 0.5),
        "mosaic": config.get("mosaic", 1.0),
        "mixup": config.get("mixup", 0.0),
        "copy_paste": config.get("copy_paste", 0.0),
        "device": config.get("device", 0),
        "workers": config.get("workers", 8),
        "cache": config.get("cache", False),
        "amp": config.get("amp", True),
        "half": config.get("half", False),
        "project": project_dir,
        "name": config.get("name", "game_char_train"),
        "exist_ok": config.get("exist_ok", False),
        "pretrained": config.get("pretrained", True),
        "resume": config.get("resume", False),
        "plots": config.get("plots", True),
        "val": config.get("val", True),
        "nbs": config.get("nbs", 64),
        "close_mosaic": config.get("close_mosaic", 10),
    }

    resume_path = config.get("resume", False)
    if resume_path and isinstance(resume_path, str) and Path(resume_path).exists():
        train_args["resume"] = True
        model = YOLO(resume_path)

    logger.info("训练参数:")
    for k, v in train_args.items():
        logger.info(f"  {k}: {v}")

    logger.info("\n开始训练...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.train(**train_args)

    logger.info("\n训练完成！")
    logger.info(f"最佳模型: {results.save_dir / 'weights' / 'best.pt'}")

    eval_results = model.val()
    logger.info(f"验证结果:")
    if hasattr(eval_results, "box"):
        box = eval_results.box
        logger.info(f"  mAP50: {box.map50:.4f}")
        logger.info(f"  mAP50-95: {box.map:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="训练游戏角色识别 YOLOv8 模型"
    )
    parser.add_argument(
        "--config", type=str, default="config/train_config.yaml",
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--check_env", action="store_true",
        help="检查运行环境"
    )
    args = parser.parse_args()

    if args.check_env:
        check_environment()
        return

    train(args.config)


if __name__ == "__main__":
    main()
