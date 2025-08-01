from ultralytics import YOLO

model = YOLO(r"C:\Users\kinac\OneDrive\Masaüstü\backup\Codes\detect_wrapper\weights\best.pt")

print("nc      :", model.model.nc)       # sınıf sayısı
print("names   :", model.names)          # isim listesi
print("yaml    :", model.args['data'])   # eğitimdeki data.yaml yolu
print("epochs  :", model.args['epochs'])
print("imgsz   :", model.args['imgsz'])
print("hyp lr0 :", model.overrides.get('lr0'))
