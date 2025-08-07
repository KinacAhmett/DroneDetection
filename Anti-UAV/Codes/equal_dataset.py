import os
import shutil


# Value set ayırma

def copy_ir_vis_to_individual_folders(src_root, dst_root, max_frames=220):
    channels = ['infrared', 'visible']  # ikisini de kapsıyoruz

    for folder in sorted(os.listdir(src_root)):
        for channel in channels:
            src_path = os.path.join(src_root, folder, channel)
            if not os.path.isdir(src_path):
                continue

            dst_folder = os.path.join(dst_root, folder, channel)
            os.makedirs(dst_folder, exist_ok=True)

            print(f"📁 {folder}/{channel} → {dst_folder} klasörüne kopyalanıyor...")
            files = sorted([f for f in os.listdir(src_path) if f.endswith('.jpg')])[:max_frames]

            for f in files:
                base = os.path.splitext(f)[0]  # örn: visible0000
                src_img = os.path.join(src_path, f)
                src_txt = os.path.join(src_path, base + '.txt')

                dst_img = os.path.join(dst_folder, f)
                dst_txt = os.path.join(dst_folder, base + '.txt')

                shutil.copy(src_img, dst_img)
                if os.path.exists(src_txt):
                    shutil.copy(src_txt, dst_txt)
                else:
                    print(f"⚠️ Etiket bulunamadı: {src_txt}")

# Yolları ayarla
src_root = r"C:\Users\kinac\DroneDetection\Anti-UAV\data\Anti-UAV-RGBT\val"
dst_root = r"C:\Users\kinac\DroneDetection\Anti-UAV\data\Anti-UAV-RGBT\val_subset_220"

copy_ir_vis_to_individual_folders(src_root, dst_root, max_frames=220)
