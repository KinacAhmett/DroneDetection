import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MultimodalYOLODataset(Dataset):
    def __init__(self, root_dir, img_size=640, repeat_ir=3, transforms=None):
        self.samples = []
        root = Path(root_dir)
        for scene in sorted(root.iterdir()):
            ir_dir  = scene / "infrared"
            vis_dir = scene / "visible"
            if not ir_dir.is_dir() or not vis_dir.is_dir():
                continue
            for ir_path in sorted(ir_dir.glob("*.jpg")):
                stem = ir_path.stem
                vis_name = stem.replace("infrared", "visible") + ".jpg"
                vis_path = vis_dir / vis_name
                label_path = ir_dir / f"{stem}.txt"
                if vis_path.exists() and label_path.exists():
                    self.samples.append((str(vis_path), str(ir_path), str(label_path)))

        assert self.samples, f"Hiç örnek bulunamadı: {root_dir}"
        self.img_size = img_size
        self.repeat_ir = repeat_ir
        self.transform = transforms if transforms else T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, ir_path, label_path = self.samples[idx]
        rgb_img = Image.open(rgb_path).convert("RGB")
        ir_img  = Image.open(ir_path).convert("L")

        rgb_tensor = self.transform(rgb_img)
        ir_tensor  = self.transform(ir_img)
        img_6ch = torch.cat([rgb_tensor, ir_tensor.repeat(self.repeat_ir,1,1)], dim=0)

        # Etiket okuma
        boxes = []
        with open(label_path, "r") as f:
            for line in f:
                values = line.strip().split()
                if len(values) == 5:
                    cls, x, y, w, h = map(float, values)
                    boxes.append([cls, x, y, w, h])
        targets = torch.tensor(boxes, dtype=torch.float32)

        return img_6ch, targets
