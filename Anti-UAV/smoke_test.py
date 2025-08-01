# smoke_test.py
import cv2
from Codes.detect_wrapper.Detectoruav import DroneDetection

# 1) Test videodan bir kare alıp kaydet
cap = cv2.VideoCapture('Codes/testvideo/testvideo/ir1.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)   # 200. kare
ret, frame = cap.read()

if ret:
    # RGB örnek
    cv2.imwrite('Codes/detect_wrapper/sample_rgb.jpg', frame)
    # IR örnek (grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('Codes/detect_wrapper/sample_ir.png', gray)
cap.release()

# 2) Modeli başlat ve iki örnek üzerinde bbox testi yap
det = DroneDetection(None, None)

# RGB testi
rgb = cv2.imread('Codes/detect_wrapper/sample_rgb.jpg')
print('RGB bbox:', det.forward_RGB(rgb))

# IR testi
ir = cv2.imread('Codes/detect_wrapper/sample_ir.png', cv2.IMREAD_GRAYSCALE)
print(' IR bbox:', det.forward_IR(ir))

