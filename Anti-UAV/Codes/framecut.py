# coding=utf-8

import os
import cv2

def video2jpg(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True
    print('read video in', video_path)
    videoclass = os.path.basename(video_path).split('.')[0]
    save_jpg_path = os.path.join(os.path.dirname(video_path), videoclass)
    if not os.path.exists(save_jpg_path):
        os.makedirs(save_jpg_path)
        print('make dir {}'.format(save_jpg_path))

    while success:
        success, frame = cap.read()
        if not success:
            break


        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        out_path = os.path.join(save_jpg_path,
                                f"{videoclass}I{str(frame_count).zfill(4)}.jpg")
        cv2.imwrite(out_path, frame, params)
        frame_count += 1

    cap.release()

def get_all_dir(root_dir):
    video_paths = []
    for entry in os.listdir(root_dir):
        path = os.path.join(root_dir, entry)
        if os.path.isdir(path):
            video_paths.extend(get_all_dir(path))
        elif path.lower().endswith('.mp4') and 'infrared' in entry.lower():
            video_paths.append(path)
    return video_paths

if __name__ == "__main__":
    root_dir = r"C:\Users\kinac\DroneDetection\Anti-UAV\data\Anti-UAV-RGBT\val"
    video_paths = get_all_dir(root_dir)
    for video_path in video_paths:
        video2jpg(video_path)
