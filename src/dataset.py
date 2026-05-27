"""
游戏角色识别数据集工具
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class GameCharacterDataset:
    """
    游戏角色数据集管理类
    用于验证、统计和分析 YOLO 格式数据集
    """

    def __init__(self, dataset_dir: str, class_names: Dict[int, str] = None):
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names or {}

        self.splits = {}
        for split_name in ["train", "val", "test"]:
            img_dir = self.dataset_dir / "images" / split_name
            lbl_dir = self.dataset_dir / "labels" / split_name
            if img_dir.exists():
                self.splits[split_name] = {
                    "images": img_dir,
                    "labels": lbl_dir if lbl_dir.exists() else None,
                }

    def get_image_label_pairs(self, split: str = "train") -> List[Tuple[Path, Path]]:
        """获取图像-标注配对列表"""
        if split not in self.splits:
            return []

        info = self.splits[split]
        pairs = []
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for img_path in info["images"].iterdir():
            if img_path.suffix.lower() in img_exts:
                if info["labels"]:
                    lbl_path = info["labels"] / (img_path.stem + ".txt")
                    if lbl_path.exists():
                        pairs.append((img_path, lbl_path))
                    else:
                        pairs.append((img_path, None))
                else:
                    pairs.append((img_path, None))

        return sorted(pairs)

    def count_instances_per_class(self, split: str = "train") -> Dict[int, int]:
        """统计每个类别的实例数量"""
        counts = {}
        for _, lbl_path in self.get_image_label_pairs(split):
            if lbl_path is None:
                continue
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        counts[cls_id] = counts.get(cls_id, 0) + 1
        return counts

    def get_statistics(self) -> dict:
        """获取数据集整体统计信息"""
        stats = {
            "total_images": 0,
            "total_instances": 0,
            "splits": {},
            "class_distribution": {},
        }

        for split_name in self.splits:
            pairs = self.get_image_label_pairs(split_name)
            stats["splits"][split_name] = len(pairs)
            stats["total_images"] += len(pairs)

            class_counts = self.count_instances_per_class(split_name)
            for cls_id, count in class_counts.items():
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                if cls_name not in stats["class_distribution"]:
                    stats["class_distribution"][cls_name] = 0
                stats["class_distribution"][cls_name] += count
                stats["total_instances"] += count

        return stats

    def validate_labels(self, split: str = "train", num_classes: int = 7) -> List[str]:
        """验证标注的正确性"""
        errors = []
        for img_path, lbl_path in self.get_image_label_pairs(split):
            if lbl_path is None:
                errors.append(f"[{img_path.name}] 缺少标注文件")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"[{img_path.name}] 无法读取图像")
                continue

            img_h, img_w = img.shape[:2]

            with open(lbl_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(parts[0])
                        if cls_id < 0 or cls_id >= num_classes:
                            errors.append(
                                f"[{img_path.name}:L{line_no}] "
                                f"类别ID {cls_id} 超出范围 [0, {num_classes - 1}]"
                            )

                        if len(parts) != 5:
                            errors.append(
                                f"[{img_path.name}:L{line_no}] "
                                f"格式错误: 需要5个值，得到 {len(parts)}"
                            )
                            continue

                        cx, cy, w, h = map(float, parts[1:])
                        if not all(0 <= v <= 1 for v in [cx, cy, w, h]):
                            errors.append(
                                f"[{img_path.name}:L{line_no}] "
                                f"坐标未归一化: [{cx}, {cy}, {w}, {h}]"
                            )

                        x1 = (cx - w / 2) * img_w
                        y1 = (cy - h / 2) * img_h
                        x2 = (cx + w / 2) * img_w
                        y2 = (cy + h / 2) * img_h
                        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                            errors.append(
                                f"[{img_path.name}:L{line_no}] "
                                f"边界框超出图像范围: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]"
                            )

                    except (ValueError, IndexError) as e:
                        errors.append(
                            f"[{img_path.name}:L{line_no}] 解析错误: {e}"
                        )

        return errors

    def print_statistics(self):
        """打印数据集统计信息"""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("数据集统计信息")
        print("=" * 60)
        print(f"总图像数: {stats['total_images']}")
        print(f"总实例数: {stats['total_instances']}")
        print(f"\n各子集图像数:")
        for split, count in stats["splits"].items():
            print(f"  {split}: {count}")

        print(f"\n各类别实例分布:")
        for cls_name, count in sorted(stats["class_distribution"].items()):
            print(f"  {cls_name}: {count}")
        print("=" * 60)
