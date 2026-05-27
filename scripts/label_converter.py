"""
标注格式转换脚本
支持: LabelMe(XML) -> YOLO,  VOC(XML) -> YOLO,  COCO(JSON) -> YOLO,  截图辅助标注
"""
import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def voc_to_yolo(
    voc_label_dir: str,
    output_dir: str,
    class_map: Dict[str, int],
    image_dir: str = None,
):
    """
    将 Pascal VOC XML 标注转换为 YOLO txt 格式

    Args:
        voc_label_dir: VOC XML 标注目录
        output_dir: YOLO txt 输出目录
        class_map: 类别名 -> ID 映射
        image_dir: 图像目录（用于获取图像尺寸，可选）
    """
    import cv2

    label_dir = Path(voc_label_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_files = list(label_dir.glob("*.xml"))
    if not xml_files:
        print(f"[WARNING] 在 {voc_label_dir} 中没有找到 XML 文件！")
        return

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename")
        if filename is not None:
            img_name = filename.text
        else:
            img_name = xml_path.stem + ".jpg"

        size = root.find("size")
        if size is not None:
            img_w = float(size.find("width").text)
            img_h = float(size.find("height").text)
        elif image_dir:
            img_path = Path(image_dir) / img_name
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img_h, img_w = img.shape[:2]
            else:
                print(f"[SKIP] 无法找到图像 {img_path}，跳过 {xml_path.name}")
                continue
        else:
            print(f"[SKIP] {xml_path.name} 缺少图像尺寸信息")
            continue

        yolo_lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in class_map:
                print(f"[WARNING] {xml_path.name}: 未知类别 '{cls_name}'，跳过")
                continue

            cls_id = class_map[cls_name]
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            yolo_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        out_path = out_dir / (xml_path.stem + ".txt")
        with open(out_path, "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"转换完成: {len(xml_files)} 个文件 -> {out_dir}")


def coco_to_yolo(
    coco_json: str,
    output_dir: str,
    image_dir: str = None,
):
    """
    将 COCO JSON 标注转换为 YOLO txt 格式

    Args:
        coco_json: COCO JSON 文件路径
        output_dir: YOLO txt 输出目录
        image_dir: 图像目录（可选）
    """
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: i for i, cat in enumerate(coco["categories"])}

    annotations_by_image: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    header = ""
    if image_dir and Path(image_dir).exists():
        header = f"image_dir: {Path(image_dir).absolute()}\n"

    for img_id, img_info in images.items():
        img_w = img_info["width"]
        img_h = img_info["height"]
        img_name = Path(img_info["file_name"]).stem

        yolo_lines = []
        for ann in annotations_by_image.get(img_id, []):
            cls_id = categories.get(ann["category_id"], -1)
            if cls_id == -1:
                continue

            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            yolo_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

        out_path = out_dir / f"{img_name}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(yolo_lines))

    cats = [cat["name"] for cat in coco["categories"]]
    print(f"COCO 转换完成: {len(images)} 张图像 -> {out_dir}")
    print(f"类别 ({len(cats)}): {cats}")

    with open(out_dir / "classes.txt", "w") as f:
        f.write("\n".join(cats))


def screenshot_label_helper(image_path: str, class_map: Dict[str, int]):
    """
    简单的手动标注辅助：命令行交互式标注
    注意: 此函数仅作参考，实际标注推荐使用 labelImg / labelme 等 GUI 工具
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"\n图像尺寸: {w}x{h}")
    print(f"可用类别: {json.dumps(class_map, ensure_ascii=False)}")
    print("\n使用 labelImg 或 labelme 进行标注会更方便。")
    print("labelImg: https://github.com/HumanSignal/labelImg")
    print("labelme: https://github.com/wkentaro/labelme")
    print("\n本项目支持 YOLO 格式标注。标注时请选择 YOLO 输出格式。")


def main():
    parser = argparse.ArgumentParser(
        description="标注格式转换工具"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    voc_parser = subparsers.add_parser("voc2yolo", help="VOC XML -> YOLO")
    voc_parser.add_argument("--label_dir", required=True, help="VOC XML 标注目录")
    voc_parser.add_argument("--output", required=True, help="YOLO 输出目录")
    voc_parser.add_argument("--class_map", required=True, help="类别映射 JSON 文件或直接 JSON 字符串")
    voc_parser.add_argument("--image_dir", default=None, help="图像目录")

    coco_parser = subparsers.add_parser("coco2yolo", help="COCO JSON -> YOLO")
    coco_parser.add_argument("--coco_json", required=True, help="COCO JSON 文件")
    coco_parser.add_argument("--output", required=True, help="YOLO 输出目录")
    coco_parser.add_argument("--image_dir", default=None, help="图像目录")

    subparsers.add_parser("label_help", help="标注工具帮助")

    args = parser.parse_args()

    if args.command == "voc2yolo":
        class_map = args.class_map
        if os.path.isfile(class_map):
            with open(class_map, "r") as f:
                class_map = json.load(f)
        else:
            class_map = json.loads(class_map)
        voc_to_yolo(args.label_dir, args.output, class_map, args.image_dir)

    elif args.command == "coco2yolo":
        coco_to_yolo(args.coco_json, args.output, args.image_dir)

    elif args.command == "label_help":
        print("推荐标注工具:")
        print("  1. labelImg - 支持 YOLO 格式输出")
        print("     https://github.com/HumanSignal/labelImg")
        print("  2. labelme - 通过本脚本转换")
        print("     https://github.com/wkentaro/labelme")
        print("  3. Roboflow - 在线标注平台")
        print("     https://roboflow.com/")
        print("\n标注时请选择 YOLO 格式（归一化中心点坐标 + 宽高）")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
