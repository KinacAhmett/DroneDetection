from utils.preprocess import extract_frames_batch, convert_labels_batch

root = "C:/Users/kinac/DroneDetection/Anti-UAV/data/Anti-UAV-RGBT/train"

extract_frames_batch(
    root,
    "C:/Users/kinac/DroneDetection/Anti-UAV/data/processed/images/train/rgb",
    "C:/Users/kinac/DroneDetection/Anti-UAV/data/processed/images/train/ir"
)

convert_labels_batch(
    root,
    "C:/Users/kinac/DroneDetection/Anti-UAV/data/processed/labels/train",
    640, 512
)
