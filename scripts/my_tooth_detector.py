"""Детектор зубов на основе своей нейросети. Без YOLO."""
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
# ВАЖНО: импортируем без префикса scripts
from my_tooth_net import MyToothNet

class MyToothDetector:
    def __init__(self, weights_path, device="cpu"):
        self.device = device
        self.net = MyToothNet().to(device)
        self.net.load_state_dict(torch.load(weights_path, map_location=device))
        self.net.eval()
        
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def predict(self, image_path):
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]

        tensor = self.transform(image=img_rgb)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.net(tensor)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        pred = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area > 200:
                boxes.append([x, y, w, h])

        return self._assign_fdi(boxes, (h_orig, w_orig))

    def _assign_fdi(self, boxes, img_shape):
        if not boxes: return []
        h_img, w_img = img_shape
        mid_x, mid_y = w_img / 2, h_img / 2
        
        upper = [b for b in boxes if (b[1] + b[3]/2) < mid_y]
        lower = [b for b in boxes if (b[1] + b[3]/2) >= mid_y]

        def number_jaw(jaw, is_upper):
            jaw.sort(key=lambda b: b[0])
            left = [b for b in jaw if (b[0] + b[2]/2) < mid_x]
            right = [b for b in jaw if (b[0] + b[2]/2) >= mid_x]
            
            q_left = 2 if is_upper else 3
            q_right = 1 if is_upper else 4
            
            left.sort(key=lambda b: b[0], reverse=True) 
            right.sort(key=lambda b: b[0])
            
            res = []
            for i, b in enumerate(left[:8]): res.append({"fdi": f"{q_left}{i+1}", "bbox": b})
            for i, b in enumerate(right[:8]): res.append({"fdi": f"{q_right}{i+1}", "bbox": b})
            return res

        return number_jaw(upper, True) + number_jaw(lower, False)