from ultralytics import YOLO, RTDETR
from pathlib import Path
import cv2
import os
import yaml
import torch


# Function to predict and save images
def predict_and_save(models, image_path, output_path, output_path_txt):
    # Perform prediction
    results = [model.predict(image_path,conf=0.5) for model in models]

    results = [result[0] for result in results]  # Get the first result from each model

    # Concatenate all boxes from all models
    all_boxes = []
    all_boxes_final_format = []
    all_scores = []
    all_classes = []

    final_boxes = []

    for res in results:
        if res.boxes is not None and len(res.boxes) > 0:
            all_boxes.append(res.boxes.xyxy)
            all_boxes_final_format.append(res.boxes.xywhn)  # normalized xywh
            all_scores.append(res.boxes.conf)
            all_classes.append(res.boxes.cls)
    if not all_boxes:
        result = results[0]  # fallback to first result if no boxes
    else:
        boxes = torch.cat(all_boxes, dim=0)
        all_boxes_final_format = torch.cat(all_boxes_final_format, dim=0)
        scores = torch.cat(all_scores, dim=0)
        classes = torch.cat(all_classes, dim=0)
        # Perform NMS
        keep = torch.ops.torchvision.nms(boxes, scores, iou_threshold=0.5)
        # Create a new result object with NMSed boxes
        result = results[0]
        # result.boxes.xyxy = boxes[keep]
        print(keep)
        for cls, xywhn, conf in zip(classes[keep], all_boxes_final_format[keep], scores[keep]):
            final_boxes.append({'cls': cls, 'xywhn': xywhn, 'conf': conf})
        
        if len(final_boxes) > 3:
            final_boxes = sorted(final_boxes, key=lambda x: float(x['conf']), reverse=True)[:3]

        # result.boxes.conf = scores[keep]
        # result.boxes.cls = classes[keep]
        # # Update normalized xywhn as well
        # xywh = (boxes[keep][:, 2:] + boxes[keep][:, :2]) / 2, boxes[keep][:, 2:] - boxes[keep][:, :2]
        # img_shape = cv2.imread(str(image_path)).shape
        # w, h = img_shape[1], img_shape[0]
        # result.boxes.xywhn = torch.cat([
        #     ((boxes[keep][:, 0] + boxes[keep][:, 2]) / 2 / w).unsqueeze(1),
        #     ((boxes[keep][:, 1] + boxes[keep][:, 3]) / 2 / h).unsqueeze(1),
        #     ((boxes[keep][:, 2] - boxes[keep][:, 0]) / w).unsqueeze(1),
        #     ((boxes[keep][:, 3] - boxes[keep][:, 1]) / h).unsqueeze(1)
        # ], dim=1)

    # Draw boxes on the image
    # img = result.plot()  # Plots the predictions directly on the image
    print(final_boxes)
    # Save the result
    # cv2.imwrite(str(output_path), img)
    # Save the bounding box data
    with open(output_path_txt, 'w') as f:
        for box in final_boxes:
            # Extract the class id and bounding box coordinates
            cls_id = int(box['cls'])
            x_center, y_center, width, height = box['xywhn'].tolist()
            
            # Write bbox information in the format [class_id, x_center, y_center, width, height]
            conf = float(box['conf'])  # confidence is a tensor with 1 value
            f.write(f"{cls_id} {conf:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


if __name__ == '__main__': 

    this_dir = Path(__file__).parent
    os.chdir(this_dir)
    with open(this_dir / "/home/ecarlson/Downloads/multi-instance-object-detection-challenge/Starter_Dataset/yolo_params.yaml", 'r') as file:
        data = yaml.safe_load(file)
        if 'test' in data and data['test'] is not None:
            images_dir = Path(data['test']) / 'images'
        else:
            print("No test field found in yolo_params.yaml, please add the test field with the path to the test images")
            exit()
    
    # check that the images directory exists
    if not images_dir.exists():
        print(f"Images directory {images_dir} does not exist")
        exit()

    if not images_dir.is_dir():
        print(f"Images directory {images_dir} is not a directory")
        exit()
    
    if not any(images_dir.iterdir()):
        print(f"Images directory {images_dir} is empty")
        exit()

    # Load the YOLO model
    detect_path = this_dir / "runs" / "detect"
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
    if len(train_folders) == 0:
        raise ValueError("No training folders found")
    # idx = 0
    # if len(train_folders) > 1:
    #     choice = -1
    #     choices = list(range(len(train_folders)))
    #     while choice not in choices:
    #         print("Select the training folder:")
    #         for i, folder in enumerate(train_folders):
    #             print(f"{i}: {folder}")
    #         choice = input()
    #         if not choice.isdigit():
    #             choice = -1
    #         else:
    #             choice = int(choice)
    #     idx = choice


    # 51 and 20
    # model_path = detect_path / train_folders[idx] / "weights" / "best.pt"

    yolo_path = detect_path / 'train51'  / "weights" / "best.pt"
    rtdetr_path = detect_path / 'train20' / "weights" / "best.pt"

    yolomodel = YOLO(yolo_path)
    rtmodel = RTDETR(rtdetr_path)

    models = [yolomodel, rtmodel]  # List of models to use for prediction, can be extended with more models if needed

    # Directory with images
    output_dir = this_dir / "predictions" # Replace with the directory where you want to save predictions
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create images and labels subdirectories
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through the images in the directory
    for img_path in images_dir.glob('*'):
        if img_path.suffix not in ['.png', '.jpg']:
            continue
        output_path_img = images_output_dir / img_path.name  # Save image in 'images' folder
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name  # Save label in 'labels' folder
        predict_and_save(models, img_path, output_path_img, output_path_txt)

    print(f"Predicted images saved in {images_output_dir}")
    print(f"Bounding box labels saved in {labels_output_dir}")
    data = this_dir / 'yolo_params.yaml'
    print(f"Model parameters saved in {data}")
    # metrics = model.val(data=data, split="test")
