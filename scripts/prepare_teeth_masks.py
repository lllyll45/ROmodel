"""Превращает YOLO-разметку зубов в маски. Рисуем эллипсы, чтобы зубы не слипались."""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
YOLO_DATA = ROOT / "data" / "dentex"
OUT_DIR = ROOT / "data" / "teeth_masks"

def convert_split(split_name: str):
    print(f"Обработка {split_name}...")
    img_dir = YOLO_DATA / split_name / "images"
    lbl_dir = YOLO_DATA / split_name / "labels"
    
    out_img_dir = OUT_DIR / split_name / "images"
    out_msk_dir = OUT_DIR / split_name / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_dir.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5: continue
                
                _, xc, yc, bw, bh = map(float, parts[:5])
                
                # Центр и радиусы (в пикселях)
                cx = int(xc * w)
                cy = int(yc * h)
                
                # УМЕНЬШАЕМ ЗУБЫ ДО 60% ОТ РАЗМЕРА РАМКИ (было 0.8)
                rx = int((bw * w) / 2 * 0.6)  
                ry = int((bh * h) / 2 * 0.6)

                # Рисуем ЗАЛИТЫЙ ЭЛЛИПС
                if rx > 0 and ry > 0:
                    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)

        cv2.imwrite(str(out_img_dir / img_path.name), img)
        cv2.imwrite(str(out_msk_dir / (img_path.stem + ".png")), mask)

if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        if (YOLO_DATA / split).exists():
            convert_split(split)
    print("Готово! Маски обновлены (использованы эллипсы).")