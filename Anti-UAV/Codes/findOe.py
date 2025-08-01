import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\kinac\DroneDetection\Anti-UAV\data\Anti-UAV-RGBT\val")

deleted = 0
for sub in ROOT.iterdir():               # train altındaki 160 klasör
    if sub.is_dir():
        ir_dir = sub / "infrared"        # ...\xxx\infrared
        if ir_dir.exists():
            shutil.rmtree(ir_dir)        # içindekilerle birlikte sil
            print(f"Silindi → {ir_dir}")
            deleted += 1

print(f"\nToplam silinen infrared klasörü: {deleted}")
