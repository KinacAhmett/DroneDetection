# train_multimodal.py

def yolo_collate(batch):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs, 0)
        return imgs, list(targets)


import os, sys, argparse, torch, csv
from detect_wrapper.utils.general import non_max_suppression, scale_coords
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
from detect_wrapper.dataset_multimodal import MultimodalYOLODataset
from detect_wrapper.models.yolo import Model
from detect_wrapper.utils.general import increment_path
from detect_wrapper.utils.torch_utils import select_device
from detect_wrapper.utils.general import ap_per_class

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',        type=str, default='models/detectx_rgb_ir.yaml')
    parser.add_argument('--data-root',  type=str, required=True)
    parser.add_argument('--val-root',   type=str)
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size',   type=int, default=640)
    parser.add_argument('--device',     type=str, default='cuda:0')
    parser.add_argument('--weights',    type=str, default='')
    parser.add_argument('--channels',   type=int, default=4)
    opt = parser.parse_args()

    # ---------------- Device & Model ----------------
    device = select_device(opt.device)
    model  = Model(opt.cfg, ch=opt.channels).to(device)
    if opt.weights and Path(opt.weights).exists():
        ckpt = torch.load(opt.weights, map_location=device)
        model.load_state_dict(ckpt['model'])

    # ---------------- Dataset & Loader --------------
    train_ds = MultimodalYOLODataset(opt.data_root, img_size=opt.img_size)
    train_loader = DataLoader(
        train_ds, batch_size=opt.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=yolo_collate)

    if opt.val_root:
        val_ds = MultimodalYOLODataset(opt.val_root, img_size=opt.img_size)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=0, pin_memory=True, collate_fn=yolo_collate)
    else:
        val_loader = None

    # ---------------- Optimizer & Loss --------------
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.937, weight_decay=5e-4)
    criterion = DetectionLoss(model)   # ✨ tam YOLO kaybı (box+obj+cls+dfl)

    # ---------------- Loggers -----------------------
    save_dir = Path(increment_path('runs/train', exist_ok=False, mkdir=True))
    (save_dir / 'tb').mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(save_dir / 'tb'))
    csv_f = open(save_dir / 'metrics.csv', 'w', newline='')
    csv_w = csv.writer(csv_f, lineterminator='\n')
    csv_w.writerow(['epoch', 'train_loss', 'map50'])

    # ---------------- Epoch Loop --------------------
    for epoch in range(opt.epochs):
        model.train(); epoch_loss = 0.0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = [t.to(device) for t in targets]

            preds = model(imgs)
            loss, loss_items = criterion(preds, torch.cat(targets))  # ✨
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            epoch_loss += loss.item()

        # ---- log train loss ----
        tb.add_scalar('train/loss', epoch_loss, epoch)
        box_l, obj_l, cls_l, dfl_l = [x.item() for x in loss_items]
        tb.add_scalars('loss_parts', {'box': box_l, 'obj': obj_l, 'cls': cls_l, 'dfl': dfl_l}, epoch)

        # ---------------- VALIDATE ------------------
        map50 = None
        if val_loader:
            model.eval(); stats = []
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(device)
                    targets = [t.to(device) for t in targets]

                    out = model(imgs)[0]
                    preds_batch = non_max_suppression(out, 0.25, 0.45)

                    for pi, pred in enumerate(preds_batch):
                        gt = targets[pi]
                        if gt.size(0):
                            gt[:, 1:] = xywh2xyxy(gt[:, 1:] * opt.img_size)  # ✨ xywh→xyxy
                        stats.append(ap_per_class(pred, gt))

            if stats:
                prec, rec, ap, f1, _ = ap_per_class.get_stats(stats)
                map50 = float(ap.mean())
                tb.add_scalar('val/mAP0.5', map50, epoch)
                print(f"[Val] Epoch {epoch+1}/{opt.epochs} | mAP@0.5 {map50:.3f}")
            model.train()

        # ---- CSV log ----
        csv_w.writerow([epoch+1, f"{epoch_loss:.4f}", f"{map50:.4f}" if map50 else ''])
        csv_f.flush(); tb.flush()

        # ---- Checkpoint ----
        if (epoch + 1) % 10 == 0:
            torch.save({'model': model.state_dict()}, save_dir / f'epoch_{epoch+1}.pt')

        print(f"[Train] Epoch {epoch+1}/{opt.epochs} — Loss: {epoch_loss:.4f}")

    csv_f.close(); tb.close()

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    train()