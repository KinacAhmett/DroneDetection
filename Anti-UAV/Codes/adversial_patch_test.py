# adversarial_patch_test.py
import cv2, argparse, os
from patch_utils import make_checker_patch, make_sine_patch, place_random_bg, place_on_bbox

# İsteğe bağlı: YOLO ile her frame tespit edip evasion’da kutu bulmak için
USE_YOLO_FOR_BBOX = False
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

def run(video_in, video_out, mode="bg_fp", patch_type="checker",
        alpha=1.0, use_yolo=False, yolo_weights=None, conf=0.25):
    cap = cv2.VideoCapture(video_in)
    assert cap.isOpened(), f"Video açılamadı: {video_in}"
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps= cap.get(cv2.CAP_PROP_FPS) or 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(video_out) or ".", exist_ok=True)
    out = cv2.VideoWriter(video_out, fourcc, fps, (W,H))

    # Patch seç
    patch = make_checker_patch(160, cells=8) if patch_type=="checker" else make_sine_patch(160, freq=14)

    model = None
    if mode=="evasion" and use_yolo:
        assert YOLO is not None and yolo_weights is not None, "YOLO yok veya weight girilmedi."
        model = YOLO(yolo_weights)

    i=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        i+=1

        if mode=="bg_fp":
            frame = place_random_bg(frame, patch, min_scale=0.08, max_scale=0.18, alpha=alpha)

        elif mode=="evasion":
            # 1) kutu biliyorsan buraya koy: bbox=(x,y,w,h)
            bbox = None
            # 2) bilmiyorsan YOLO ile bul:
            if use_yolo and model is not None:
                res = model.predict(frame, conf=conf, verbose=False)[0]
                if len(res.boxes) > 0:
                    # en yüksek conf’lu kutu
                    b = res.boxes.xywh.cpu().numpy()[0]  # cx,cy,w,h
                    cx, cy, w, h = map(int, b)
                    x = cx - w//2; y = cy - h//2
                    bbox = (x,y,w,h)
            # Kutun varsa patch’i üstüne koy
            if bbox is not None:
                frame = place_on_bbox(frame, patch, bbox, rel_scale=0.6, rot_range=20, alpha=alpha)

        out.write(frame)

    cap.release(); out.release()
    print(f"[✓] Yazıldı → {video_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="video_in",  required=True)
    ap.add_argument("--out", dest="video_out", required=True)
    ap.add_argument("--mode", choices=["bg_fp","evasion"], default="bg_fp")
    ap.add_argument("--patch", choices=["checker","sine"], default="checker")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--use_yolo", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    run(args.video_in, args.video_out, mode=args.mode, patch_type=args.patch,
        alpha=args.alpha, use_yolo=args.use_yolo, yolo_weights=args.weights, conf=args.conf)
