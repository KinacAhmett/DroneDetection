import os
import shutil

def split_half_per_folder(src_root, dst_root1, dst_root2):
    channels = ['infrared', 'visible']

    for folder in sorted(os.listdir(src_root)):
        folder_path = os.path.join(src_root, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"📁 Klasör işleniyor: {folder}")
        
        for channel in channels:
            src_channel_path = os.path.join(folder_path, channel)
            if not os.path.isdir(src_channel_path):
                print(f"  ❌ {channel} klasörü eksik: {src_channel_path}")
                continue

            files = sorted([f for f in os.listdir(src_channel_path) if f.endswith('.jpg')])
            total = len(files)
            if total == 0:
                print(f"  ⚠️ {folder}/{channel} içinde hiç jpg yok")
                continue

            half = total // 2

            dst1_path = os.path.join(dst_root1, folder, channel)
            dst2_path = os.path.join(dst_root2, folder, channel)
            os.makedirs(dst1_path, exist_ok=True)
            os.makedirs(dst2_path, exist_ok=True)

            for i, f in enumerate(files):
                base = os.path.splitext(f)[0]
                src_img = os.path.join(src_channel_path, f)
                src_txt = os.path.join(src_channel_path, base + '.txt')

                # hedef belirle
                dst_folder = dst1_path if i < half else dst2_path

                shutil.copy(src_img, os.path.join(dst_folder, f))
                if os.path.exists(src_txt):
                    shutil.copy(src_txt, os.path.join(dst_folder, base + '.txt'))
                else:
                    print(f"    ⚠️ Etiket yok: {src_txt}")

            print(f"  ✅ {channel}: {total} → {half}/{total - half} bölündü")

# 📁 Senin klasör yollarını buraya gir
src_root = r"C:\Users\kinac\DroneDetection\Anti-UAV\data\Anti-UAV-RGBT\train"
dst_root1 = r"C:\Users\kinac\DroneDetection\Anti-UAV-RGBT\train_part1"
dst_root2 = r"C:\Users\kinac\DroneDetection\Anti-UAV-RGBT\train_part2"

split_half_per_folder(src_root, dst_root1, dst_root2)
