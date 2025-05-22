# from ultralytics import RTDETR

# # # Load a COCO-pretrained RT-DETR-l model
# model = RTDETR("rtdetr-l.yaml")

# # # Display model information (optional)
# model.info()

# # # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="/home/ecarlson/Desktop/myCoco.yml", epochs=10, imgsz=640)

# from ultralytics import RTDETR

# # # Load a COCO-pretrained RT-DETR-l model
# model = RTDETR("rtdetr-l.yaml")

# # # Display model information (optional)
# model.info()

# # # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=10, imgsz=640)







from ultralytics import CustomRTDETR

# # Load a COCO-pretrained RT-DETR-l model
model = CustomRTDETR("custom-rtdetr-l.yaml")

# # Display model information (optional)
model.info()

# # Train the model on the COCO8 example dataset for 10 epochs
results = model.train(data="/home/ecarlson/Desktop/myCoco.yml", epochs=10, imgsz=640)

# from ultralytics import CustomRTDETR

# # # Load a COCO-pretrained RT-DETR-l model
# model = CustomRTDETR("custom-rtdetr-l.yaml")

# # # Display model information (optional)
# model.info()

# # # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=10, imgsz=640)