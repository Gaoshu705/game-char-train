"""
工具函数
"""
import os
import sys
import yaml
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(*paths: str) -> Path:
    """将相对路径解析为项目根目录下的绝对路径"""
    return PROJECT_ROOT.joinpath(*paths)


def load_config(config_path: str) -> Dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str = "runs/logs", name: str = "game_char_train"):
    """设置日志系统"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(
        log_path / f"{name}.log", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray = None,
    class_names: Dict[int, str] = None,
    colors: Dict[int, Tuple[int, int, int]] = None,
    conf_threshold: float = 0.25,
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    在图像上绘制检测框

    Args:
        image: BGR 图像
        boxes: [N, 4] xyxy 格式边界框
        class_ids: [N] 类别 ID
        scores: [N] 置信度（可选）
        class_names: 类别名映射
        colors: 每类颜色映射
        conf_threshold: 置信度阈值
        line_thickness: 框线厚度
        font_scale: 字体大小

    Returns:
        绘制后的图像
    """
    img = image.copy()
    h, w = img.shape[:2]

    if class_names is None:
        class_names = {}
    if colors is None:
        rng = np.random.default_rng(42)
        colors = {}
        for i in range(100):
            colors[i] = tuple(map(int, rng.integers(0, 255, 3)))

    for i in range(len(boxes)):
        if scores is not None and scores[i] < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, boxes[i])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        cls_id = int(class_ids[i])
        color = colors.get(cls_id, (0, 255, 0))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        label_parts = []
        if class_names and cls_id in class_names:
            label_parts.append(class_names[cls_id])
        else:
            label_parts.append(f"cls_{cls_id}")

        if scores is not None:
            label_parts.append(f"{scores[i]:.2f}")

        label = " ".join(label_parts)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, line_thickness // 2)
        )

        cv2.rectangle(
            img,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, line_thickness // 2),
        )

    return img


def visualize_predictions(
    image_path: str,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray = None,
    class_names: Dict[int, str] = None,
    output_path: str = None,
    conf_threshold: float = 0.25,
):
    """
    可视化预测结果并保存

    Args:
        image_path: 输入图像路径
        boxes: 检测框 [N, 4] xyxy
        class_ids: 类别ID [N]
        scores: 置信度 [N]
        class_names: 类别名映射
        output_path: 输出路径
        conf_threshold: 置信度阈值
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] 无法读取图像: {image_path}")
        return None

    result = draw_boxes(img, boxes, class_ids, scores, class_names, conf_threshold=conf_threshold)

    if output_path:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        print(f"[INFO] 结果已保存到: {output_path}")

    return result


def get_device(device: Union[str, int] = 0) -> torch.device:
    """获取可用设备"""
    if device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_environment():
    """检查运行环境"""
    print("\n" + "=" * 50)
    print("运行环境检查")
    print("=" * 50)

    print(f"Python: {sys.version}")

    try:
        import ultralytics
        print(f"ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("ultralytics: [未安装] 请运行: pip install ultralytics")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch: [未安装]")

    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except ImportError:
        print("OpenCV: [未安装]")

    print("=" * 50)
