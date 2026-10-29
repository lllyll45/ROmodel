"""Обучение своей сети для поиска зубов."""
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import segmentation_models_pytorch as smp

from my_tooth_net import MyToothNet

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "teeth_masks"

class TeethDataset(Dataset):
    def __init__(self, split, transform):
        self.img_dir = DATA_DIR / split / "images"
        self.msk_dir = DATA_DIR / split / "masks"
        self.files = sorted(self.img_dir.glob("*.jpg"))
        self.transform = transform
        
    def __len__(self): 
        return len(self.files)
        
    def __getitem__(self, idx):
        img_path = self.files[idx]
        msk_path = self.msk_dir / (img_path.stem + ".png")
        
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        
        aug = self.transform(image=img, mask=msk)
        return aug["image"], aug["mask"].long()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--name", type=str, default="default_run")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_transform = A.Compose([
        A.Resize(args.imgsz, args.imgsz),
        A.HorizontalFlip(p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.5), # <-- ДОБАВЛЕНО ДЛЯ КОНТРАСТА
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_ds = TeethDataset("train", train_transform)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    model = MyToothNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode="multiclass")
    
    def criterion(pred, target):
        return ce_loss(pred, target) + dice_loss(pred, target)

    save_path = ROOT / "best_model" / f"my_tooth_net_{args.name}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for imgs, msks in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, msks = imgs.to(device), msks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, msks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:02d}: loss = {avg_loss:.4f}")

        torch.save(model.state_dict(), save_path)
        
    print("\nОбучение завершено!")

if __name__ == "__main__":
    main()