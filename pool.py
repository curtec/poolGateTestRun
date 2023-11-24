from ultralytics import YOLO

model = YOLO("yolov8x.yaml")
model = YOLO("yolov8x.pt")


results = model.train(data="coco128.yaml", epochs=1)
results = model.val()
results = model("https://ultralytics.com/images/bus.jpg")
success = model.export(format="onnx")

