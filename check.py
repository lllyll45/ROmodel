import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts")) # <-- ДОБАВИТЬ ЭТУ СТРОКУ

from scripts.my_tooth_detector import MyToothDetector
import cv2

# Путь к тестовому снимку 
img_path = Path("data/dentex/test/images/train_13.jpg") 

# ЗАМЕНИТЕ имя файла на победителя из sweep (например, my_tooth_net_lr_1e3.pth)
weights_path = "best_model/my_tooth_net_my_net_512_lr1e4.pth" 

# Загружаем нашу модель
detector = MyToothDetector(weights_path)

# Получаем список зубов
teeth = detector.predict(img_path)

print(f"Найдено зубов: {len(teeth)}")
for t in teeth:
    print(f"Зуб {t['fdi']}: рамка {t['bbox']}")

# Отрисовка (чтобы посмотреть глазами)
img = cv2.imread(str(img_path))
for t in teeth:
    x, y, w, h = t['bbox']
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, str(t['fdi']), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite("result31_4.jpg", img)
print("Результат сохранен в result.jpg")