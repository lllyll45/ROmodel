

import argparse
import shutil
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]


class DentexPrepareService:
    """Скачивает DENTEX enumeration и раскладывает снимки в формат YOLO."""

    _DATASET_DIR = ROOT / "data" / "dentex"
    _TOOTH_CLASS_ID = 0
    _SPLIT_DIRS = {
        "train": "train",
        "validation": "valid",
        "test": "test",
    }

    def prepare(self, force: bool = False) -> None:
        if self._already_prepared() and not force:
            print(f"Датасет уже лежит в {self._DATASET_DIR}. Для пересборки: --force")
            return

        print("Качаю DENTEX enumeration...")
        dataset = load_dataset("tirandazdylan/dentex-panoramic", "enumeration")
        total_boxes = sum(self._write_split(name, dataset[name]) for name in self._SPLIT_DIRS)
        self._write_yaml()
        print(f"Готово. Рамок: {total_boxes}. Описание: {self._DATASET_DIR / 'data.yaml'}")

    def _already_prepared(self) -> bool:
        yaml_path = self._DATASET_DIR / "data.yaml"
        images = self._DATASET_DIR / "train" / "images"
        if not yaml_path.is_file() or not any(images.glob("*.jpg")):
            return False
        text = yaml_path.read_text(encoding="utf-8")
        return "nc: 1" in text and "tooth" in text

    def _write_split(self, split_name: str, rows) -> int:
        folder = self._SPLIT_DIRS[split_name]
        self._clear_split(folder)
        image_dir = self._DATASET_DIR / folder / "images"
        label_dir = self._DATASET_DIR / folder / "labels"
        seen: set[str] = set()
        box_total = 0
        for index, row in enumerate(rows, start=1):
            image_id = self._image_id(row, index, seen)
            row["image"].save(image_dir / f"{image_id}.jpg", format="JPEG", quality=95)
            box_total += self._write_label(label_dir / f"{image_id}.txt", row)
        print(f"{folder}: {len(seen)} снимков, {box_total} рамок")
        return box_total

    def _image_id(self, row: dict, index: int, seen: set[str]) -> str:
        image_id = str(row["image_id"])
        if image_id in seen:
            image_id = f"{image_id}_{index}"
        seen.add(image_id)
        return image_id

    def _clear_split(self, folder: str) -> None:
        for name in ("images", "labels"):
            path = self._DATASET_DIR / folder / name
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

    def _write_label(self, label_path: Path, row: dict) -> int:
        width = int(row["width"])
        height = int(row["height"])
        lines = [self._label_line(bbox, width, height) for bbox in row["objects"]["bbox"]]
        lines = [line for line in lines if line]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(lines)

    def _label_line(self, bbox, width: int, height: int) -> str:
        left, top, box_width, box_height = (float(value) for value in bbox)
        if box_width <= 0 or box_height <= 0 or width <= 0 or height <= 0:
            return ""
        x_center = self._clip((left + box_width / 2) / width)
        y_center = self._clip((top + box_height / 2) / height)
        return (
            f"{self._TOOTH_CLASS_ID} {x_center:.6f} {y_center:.6f} "
            f"{self._clip(box_width / width):.6f} {self._clip(box_height / height):.6f}"
        )

    def _write_yaml(self) -> None:
        content = """train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: tooth
"""
        (self._DATASET_DIR / "data.yaml").write_text(content, encoding="utf-8")

    def _clip(self, value: float) -> float:
        return min(1.0, max(0.0, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать DENTEX enumeration и разложить в формат YOLO")
    parser.add_argument("--force", action="store_true", help="перезаписать уже скачанные снимки")
    args = parser.parse_args()
    DentexPrepareService().prepare(force=args.force)


if __name__ == "__main__":
    main()