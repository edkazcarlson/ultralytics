# TODO
#  Run a 50-100 epoch train for VOC for both the default rtdetr and the custom rtdetr
#  Once I can confirm custom rtdetr works, then update CustomDeformableTransformerDecoder to do an extra head on SOME of the registers to predict loss, then update the CustomDetrLoss to calculate the loss prediction loss, based on the epoch entry in the batch. Also update the logic in the aux loss so that we no longer read from every single layer, but just the last layer per lap.


# Points to change to do loss loss are:



# from ultralytics import RTDETR

# # # Load a COCO-pretrained RT-DETR-l model
# model = RTDETR("rtdetr-l.yaml")

# # # Display model information (optional)
# model.info()

# # # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="/home/ecarlson/Desktop/myVoc.yml", epochs=50, imgsz=640)
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
results = model.train(data="/home/ecarlson/Desktop/myVoc.yml", epochs=100, imgsz=640)
# results = model.train(data="/home/ecarlson/Desktop/myCoco.yml", epochs=10, imgsz=640)




# from ultralytics import CustomRTDETR

# # # Load a COCO-pretrained RT-DETR-l model
# model = CustomRTDETR("custom-rtdetr-l.yaml")

# # # Display model information (optional)
# model.info()

# # # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=10, imgsz=640)