from ultralytics import YOLO

model = YOLO("best_80.pt")

model.predict(source="helmetvid1.mp4", conf=0.2, save=True)
