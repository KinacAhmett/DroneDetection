import os
import cv2
import json
from tqdm import tqdm

def extract_frames_batch(root_dir, out_rgb_dir, out_ir_dir):
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_ir_dir, exist_ok=True)
    
    folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for folder in tqdm(folders, desc="Extracting frames"):
        rgb_path = os.path.join(folder, 'visible.mp4')
        ir_path = os.path.join(folder, 'infrared.mp4')

        video_name = os.path.basename(folder)

        extract(rgb_path, out_rgb_dir, f"rgb_{video_name}")
        extract(ir_path, out_ir_dir, f"ir_{video_name}")


def extract(video_path, output_dir, prefix):
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = f"{prefix}_{i:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        i += 1
    cap.release()

def convert_labels_batch(root_dir, out_label_dir, img_w, img_h):
    os.makedirs(out_label_dir, exist_ok=True)
    
    folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for folder in tqdm(folders, desc="Converting labels"):
        json_path = os.path.join(folder, 'infrared.json')
        video_name = os.path.basename(folder)

        with open(json_path) as f:
            data = json.load(f)

        for i, bbox in enumerate(data['gt_rect']):
            label_file = os.path.join(out_label_dir, f"frame_{video_name}_{i:04d}.txt")

            if data['exist'][i] == 0 or not bbox or len(bbox) != 4:
                open(label_file, 'w').close()
                continue

            x, y, w, h = bbox
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            with open(label_file, 'w') as f_txt:
                f_txt.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
