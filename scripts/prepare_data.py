"""
数据集准备脚本
功能: 将原始标注和数据转换为YOLO格式，并划分训练/验证/测试集
从 data/raw/labels/classes.txt 读取类别并自动同步到 data/dataset.yaml
"""
import os
import sys
import shutil
import random
import argparse
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import resolve_path


def sync_classes_to_dataset_yaml(
    raw_labels_dir: str,
    dataset_yaml_path: str = "data/dataset.yaml",
):
    """
    从 raw_labels_dir/classes.txt 读取类别列表，
    同步写入 dataset.yaml 的 names 和 nc 字段

    Args:
        raw_labels_dir: 原始标注目录（内含 classes.txt）
        dataset_yaml_path: 数据集配置文件路径（相对于项目根）

    Returns:
        (num_classes, class_names) 元组
    """
    labels_dir = Path(raw_labels_dir)
    classes_file = labels_dir / "classes.txt"

    if not classes_file.exists():
        print(f"[WARNING] 未找到 {classes_file}，跳过类别同步。")
        print(f"  请在 {raw_labels_dir} 下创建 classes.txt，每行一个类别名。")
        return None, {}

    with open(classes_file, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    if not class_names:
        print(f"[WARNING] {classes_file} 为空，跳过类别同步。")
        return None, {}

    num_classes = len(class_names)
    names_dict = {i: name for i, name in enumerate(class_names)}

    dataset_yaml = resolve_path(dataset_yaml_path)

    if dataset_yaml.exists():
        with open(dataset_yaml, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    config["names"] = names_dict
    config["nc"] = num_classes

    if "path" not in config:
        config["path"] = str(resolve_path("data/dataset").as_posix())
    if "train" not in config:
        config["train"] = "images/train"
    if "val" not in config:
        config["val"] = "images/val"
    if "test" not in config:
        config["test"] = "images/test"

    with open(dataset_yaml, "w", encoding="utf-8") as f:
        f.write("# YOLOv8 数据集配置文件\n")
        f.write("# 游戏角色识别与区分\n\n")
        f.write(f"# 数据集根路径\npath: {config['path']}\n\n")
        f.write("# 训练/验证/测试集路径（相对于 path）\n")
        f.write(f"train: {config['train']}\n")
        f.write(f"val: {config['val']}\n")
        f.write(f"test: {config['test']}\n\n")
        f.write("# 类别定义（由 classes.txt 自动生成）\n")
        f.write("names:\n")
        for cls_id, cls_name in sorted(names_dict.items()):
            f.write(f"  {cls_id}: {cls_name}\n")
        f.write(f"\n# 类别数量\nnc: {num_classes}\n")

    print(f"\n类别已同步到 {dataset_yaml_path}:")
    print(f"  类别数: {num_classes}")
    print(f"  类别列表:")
    for cls_id, cls_name in sorted(names_dict.items()):
        print(f"    {cls_id}: {cls_name}")

    return num_classes, names_dict


def split_dataset(
    raw_images: str,
    raw_labels: str,
    output_dir: str,
    splits: tuple = (0.7, 0.2, 0.1),
    seed: int = 42,
):
    """
    将原始数据按比例划分为 train/val/test

    Args:
        raw_images: 原始图像目录
        raw_labels: 原始YOLO标注目录
        output_dir: 输出目录（data/dataset）
        splits: 训练/验证/测试比例
        seed: 随机种子
    """
    images_dir = Path(raw_images)
    labels_dir = Path(raw_labels)
    out = Path(output_dir)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_exts
    ])

    if not image_files:
        print(f"[WARNING] 在 {raw_images} 中没有找到图像文件！")
        return

    random.seed(seed)
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])

    subsets = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:],
    }

    for subset_name, files in subsets.items():
        img_out = out / "images" / subset_name
        lbl_out = out / "labels" / subset_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            shutil.copy2(img_path, img_out / img_path.name)

            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, lbl_out / label_path.name)

        print(f"[{subset_name}] {len(files)} 张图像")


def check_dataset(output_dir: str, num_classes: int = 7):
    """
    检查数据集的完整性

    Args:
        output_dir: 数据集输出目录
        num_classes: 类别总数
    """
    out = Path(output_dir)
    issues = []

    for subset in ["train", "val", "test"]:
        img_dir = out / "images" / subset
        lbl_dir = out / "labels" / subset

        if not img_dir.exists():
            issues.append(f"缺失目录: {img_dir}")
            continue

        images = set(f.stem for f in img_dir.iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        labels = set(f.stem for f in lbl_dir.iterdir()
                     if f.suffix == ".txt") if lbl_dir.exists() else set()

        orphan_images = images - labels
        orphan_labels = labels - images

        if orphan_images:
            issues.append(
                f"[{subset}] {len(orphan_images)} 张图像缺少标注"
            )

        if orphan_labels:
            issues.append(
                f"[{subset}] {len(orphan_labels)} 个标注缺少图像"
            )

        for lbl_stem in labels & images:
            lbl_file = lbl_dir / f"{lbl_stem}.txt"
            with open(lbl_file, "r") as f:
                for line_no, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(parts[0])
                        if cls_id < 0 or cls_id >= num_classes:
                            issues.append(
                                f"[{subset}] {lbl_stem}.txt 行{line_no}: "
                                f"类别ID {cls_id} 超出范围 [0, {num_classes - 1}]"
                            )
                        coords = list(map(float, parts[1:]))
                        if len(coords) != 4:
                            issues.append(
                                f"[{subset}] {lbl_stem}.txt 行{line_no}: "
                                f"需要有4个坐标值，得到 {len(coords)}"
                            )
                        if any(not (0 <= v <= 1) for v in coords):
                            issues.append(
                                f"[{subset}] {lbl_stem}.txt 行{line_no}: "
                                f"坐标应归一化到 [0, 1]，得到 {coords}"
                            )
                    except ValueError:
                        issues.append(
                            f"[{subset}] {lbl_stem}.txt 行{line_no}: 格式错误"
                        )

    if issues:
        print("\n检测到以下问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n数据集检查通过，未发现问题！")

    print(f"\n数据集统计:")
    for subset in ["train", "val", "test"]:
        img_dir = out / "images" / subset
        if img_dir.exists():
            count = len(list(img_dir.iterdir()))
            print(f"  {subset}: {count} 张图像")


def main():
    parser = argparse.ArgumentParser(
        description="准备YOLO数据集：划分train/val/test并检查"
    )
    parser.add_argument(
        "--raw_images", type=str, default="data/raw/images",
        help="原始图像目录"
    )
    parser.add_argument(
        "--raw_labels", type=str, default="data/raw/labels",
        help="原始YOLO标注目录"
    )
    parser.add_argument(
        "--output", type=str, default="data/dataset",
        help="输出数据集目录"
    )
    parser.add_argument(
        "--splits", type=float, nargs=3, default=[0.7, 0.2, 0.1],
        help="train/val/test 划分比例"
    )
    parser.add_argument(
        "--num_classes", type=int, default=None,
        help="类别数量（留空则从 data/raw/labels/classes.txt 自动推断）"
    )
    parser.add_argument(
        "--check_only", action="store_true",
        help="仅检查数据集，不进行划分"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )
    args = parser.parse_args()

    raw_labels_dir = str(resolve_path(args.raw_labels))
    output_dir = str(resolve_path(args.output))

    num_classes, names_dict = sync_classes_to_dataset_yaml(raw_labels_dir)

    if num_classes is None:
        num_classes = args.num_classes or 1
    elif args.num_classes is not None and args.num_classes != num_classes:
        print(
            f"[WARNING] --num_classes={args.num_classes} 与 "
            f"classes.txt 检测到的 {num_classes} 不一致，以 classes.txt 为准"
        )

    if args.check_only:
        check_dataset(output_dir, num_classes)
    else:
        split_dataset(
            str(resolve_path(args.raw_images)),
            raw_labels_dir,
            output_dir,
            tuple(args.splits),
            args.seed,
        )
        check_dataset(output_dir, num_classes)


if __name__ == "__main__":
    main()
