# convert_json_to_yolo.py
import json
import cv2
from pathlib import Path


def convert_scene(scene_dir: Path, modality: str):
    """
    scene_dir/
      visible.json, infrared.json
      visible/visibleI0000.jpg ...
      infrared/infraredI0000.jpg ...
    """
    jfile = scene_dir / f"{modality}.json"
    img_dir = scene_dir / modality
    if not (jfile.exists() and img_dir.exists()):
        print(f"⏩ {jfile} ya da {img_dir} yok, atlandı.")
        return

    data   = json.loads(jfile.read_text('utf-8'))
    rects  = data.get("gt_rect", [])
    exists = data.get("exist", [])

    n = min(len(rects), len(exists))
    for idx in range(n):
        img_name  = f"{modality}I{idx:04d}.jpg"
        img_path  = img_dir / img_name
        if not img_path.exists():
            img_path = img_path.with_suffix('.png')
            if not img_path.exists():
                continue  # kare gerçekten yok

        label_path = img_path.with_suffix('.txt')

        # ------------ Drone yoksa boş .txt ------------
        if exists[idx] != 1 or not rects[idx] or len(rects[idx]) != 4:
            label_path.write_text('')
            continue

        # ------------ Bbox bilgisi ------------
        x, y, bw, bh = rects[idx]

        # ------------ Kare boyutu (gerçek) ------------
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        # ------------ Resim sınırı içinde kırp ------------
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + bw)
        y2 = min(H, y + bh)
        if x2 <= x1 or y2 <= y1:
            label_path.write_text('')        # tamamen dışarıdaysa atla
            continue

        bw = x2 - x1
        bh = y2 - y1
        xc = x1 + bw/2
        yc = y1 + bh/2

        # ------------ YOLO normalize ------------
        out = f"0 {xc/W:.6f} {yc/H:.6f} {bw/W:.6f} {bh/H:.6f}\n"
        label_path.write_text(out)

def main(root_dir):
    root = Path(root_dir)
    scenes = [p for p in root.iterdir() if p.is_dir()]
    for scene in scenes:
        convert_scene(scene, "infrared")
    print("✅ Tüm JSON dosyaları YOLO‑txt formatına dönüştürüldü.")

if __name__ == "__main__":
    # Kök klasörü buraya yaz
    main(r"C:\Users\kinac\DroneDetection\Anti-UAV\data\Anti-UAV-RGBT\val")
