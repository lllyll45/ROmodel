"""Перебор нескольких настроек для своей сети MyToothNet."""
import argparse
import csv
import shutil
import subprocess
import sys
import time
import re
from datetime import datetime
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "weights" / "candidates"
SUMMARY_PATH = CANDIDATES / "summary.csv"
REPORT_PATH = CANDIDATES / "report.md"

# 5 вариантов: меняем learning rate, размер картинки и батч
EXPERIMENTS = (
    {"name": "my_net_512_lr1e4", "imgsz": 512, "batch": 8, "lr": 1e-4},
    {"name": "my_net_512_lr3e4", "imgsz": 512, "batch": 8, "lr": 3e-4},
    {"name": "my_net_512_lr1e5", "imgsz": 512, "batch": 8, "lr": 1e-5},
    {"name": "my_net_384_lr1e4", "imgsz": 384, "batch": 8, "lr": 1e-4},
    {"name": "my_net_512_lr1e3", "imgsz": 512, "batch": 8, "lr": 1e-3},
)

COLUMNS = ("name", "imgsz", "batch", "lr", "epoch", "final_loss", "weights", "status")

def _parse_args():
    parser = argparse.ArgumentParser()
    # СТАВИМ 30 ЭПОХ ПО УМОЛЧАНИЮ
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--only", default="", help="имя одного варианта")
    parser.add_argument("--force", action="store_true", help="переобучить даже если веса есть")
    return parser.parse_args()

def _render_report(records, sweep_done):
    finished = [r for r in records if r["status"] == "ok"]
    finished.sort(key=lambda r: float(r["final_loss"]))
    
    lines = [
        "# Отчёт перебора гиперпараметров (MyToothNet)",
        "",
        f"Обновлён: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Цифры — loss с последней эпохи (после {records[0]['epoch'] if records else '?'} эпох обучения).",
        "",
        "| место | вариант | imgsz | batch | lr | эпоха | final_loss | статус |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, row in enumerate(finished, start=1):
        lines.append(
            f"| {i} | {row['name']} | {row['imgsz']} | {row['batch']} | {row['lr']} | "
            f"{row['epoch']} | {row['final_loss']} | ok |"
        )
    lines.append("")
    if finished:
        lines.append(f"**Лучший вариант:** `{finished[0]['name']}` (loss={finished[0]['final_loss']}).")
    return "\n".join(lines)

def main():
    args = _parse_args()
    
    if not torch.cuda.is_available():
        print("ВНИМАНИЕ: GPU не найден. Перебор на CPU займет недели.")

    selected = [item for item in EXPERIMENTS if not args.only or item["name"] == args.only]
    records = []
    
    CANDIDATES.mkdir(parents=True, exist_ok=True)

    for exp in selected:
        name = exp["name"]
        destination = CANDIDATES / f"{name}.pth"
        
        if destination.is_file() and not args.force:
            print(f"Пропускаю {name}: веса уже есть.")
            continue

        print(f"\n" + "="*50)
        print(f"Старт {name}: imgsz={exp['imgsz']} batch={exp['batch']} lr={exp['lr']} (эпох: {args.epochs})")
        print("="*50)
        
        cmd = [
            sys.executable, "scripts/train_my_net.py",
            "--imgsz", str(exp["imgsz"]),
            "--batch", str(exp["batch"]),
            "--lr", str(exp["lr"]),
            "--epochs", str(args.epochs),
            "--name", name
        ]
        
        start_time = time.time()
        process = subprocess.run(cmd, cwd=ROOT)
        
        output = process.stdout + process.stderr
        losses = re.findall(r"Epoch \d+: loss = ([\d.]+)", output)
        
        if losses:
            final_loss = float(losses[-1])
            status = "ok"
            src_weights = ROOT / "best_model" / f"my_tooth_net_{name}.pth"
            if src_weights.is_file():
                shutil.copy2(src_weights, destination)
            records.append({
                "name": name, "imgsz": str(exp["imgsz"]), "batch": str(exp["batch"]),
                "lr": str(exp["lr"]), "epoch": str(args.epochs), 
                "final_loss": f"{final_loss:.4f}", "weights": str(destination), "status": "ok"
            })
            print(f"✅ Завершено {name}: loss = {final_loss:.4f}")
        else:
            print(f"❌ Ошибка при запуске {name}")
            records.append({
                "name": name, "imgsz": str(exp["imgsz"]), "batch": str(exp["batch"]),
                "lr": str(exp["lr"]), "epoch": "0", 
                "final_loss": "inf", "weights": "", "status": "error"
            })

    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
        
    REPORT_PATH.write_text(_render_report(records, True), encoding="utf-8")
    print(f"\nОтчет сохранен: {REPORT_PATH}")

if __name__ == "__main__":
    main()