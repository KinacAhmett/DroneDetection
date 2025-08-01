import cv2
import numpy as np
import torch
from ultralytics import YOLO
from .utils.torch_utils import select_device

class DroneDetection:
    def __init__(self, IRweights_path, RGBweights_path=None, *args, **kwargs):
        """
        Sadece model yüklemesi yapıyor.
        """
        # GPU/CPU seçimi
        self.device = select_device('')  # otomatik CUDA yoksa CPU
        torch.backends.cudnn.benchmark = True

        self.yolo = YOLO(IRweights_path)
        self.conf_thresh = 0.05   # confidences eşiği
        self.iou_thresh  = 0.20
        # YOLO11 nano modelini indir ve yükle (COCO ön-eğitimli)
        """
        self.yolo = YOLO('yolo11n.pt')
        """


    def forward_IR(self, frame: np.ndarray):
        if frame is None:
            raise ValueError("IR frame None!")
    # Eğer 3 kanallıysa, doğrudan kullan; tek kanallıysa BGR'ye çevir
        if len(frame.shape) == 2:
            ir_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            ir_rgb = frame
        else:
            raise ValueError("IR frame formatı beklenenden farklı!")
    # ...devamı...
        
        results = self.yolo.predict(
           source=ir_rgb,
           conf=self.conf_thresh,
           iou=self.iou_thresh,
           verbose=False)
        
        if results and len(results[0].boxes) > 0:
            x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy()
            return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        return None

    def forward_RGB(self, frame: np.ndarray):
        """
        RGB görüntüyü doğrudan predict ediyor.
        """
        results = self.yolo.predict(source=frame, conf=0.10, iou=0.30, verbose=False)
        
        results = self.yolo.predict(
           source=frame,
           conf=self.conf_thresh,
           iou=self.iou_thresh,
           verbose=False
       )
        if results and len(results[0].boxes) > 0:
            x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy()
            return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        return None

if __name__ == '__main__':
    # Smoke-test kodu
    det = DroneDetection()
    rgb = cv2.imread('Codes/detect_wrapper/sample_rgb.jpg')
    print('RGB bbox:', det.forward_RGB(rgb))
    ir = cv2.imread('Codes/detect_wrapper/sample_ir.png', cv2.IMREAD_GRAYSCALE)
    print(' IR bbox:', det.forward_IR(ir))
