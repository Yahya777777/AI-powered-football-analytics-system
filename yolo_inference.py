from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict('input/rca_vs_mas_test.mp4', save=True, project='runs')
print(results[0])
print('-----------------------------------------------')
for box in results[0].boxes:
    print(box)