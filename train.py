# EPOCHS = 1
# EPOCHS = 35
EPOCHS = 40 # 0.97593,0.95095 WITHOUT THE curve fitting transform 
                # 0.9 temp on curve fitting:
                #    all        208        454      0.984      0.939      0.978       0.95

                # 0.75 temp
                #    all        208        454      0.992      0.932       0.97      0.949


        # img = self.brightenDarkness(img, 0.2, meanBlur=True)
        # img = img ** (0.9 + (img *0.09))
        # img = self.brightenDarkness(img, 0.15)
        # img = img ** (0.9 + (img *0.09))

                #    all        208        454      0.997      0.934      0.974      0.951

# EPOCHS = 100 #                   all         37         81      0.996      0.951      0.992      0.977
# EPOCHS = 10
MOSAIC = 0.4
OPTIMIZER = 'AdamW'
MOMENTUM = 0.9
LR0 = 0.0001
LRF = 0.0001
SINGLE_CLS = True
import argparse
from ultralytics import YOLO
import os
import sys

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    # mosaic
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    # optimizer
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    # momentum
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    # lr0
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    # lrf
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    # single_cls
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    args = parser.parse_args()
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    # model = YOLO("yolo11x.pt") # 0.987      0.962 training on val+train of my data  0.976      0.958 w/ my training on my train set , 0.985   0.961  w/o any of my data.    NO CHANGE BASELINE: 0.98 0.961
    model = YOLO("yolo12x.pt")  # train+val of my data + lens changes: 0.978      0.961   My data + lens changes: 0.99  0.973  No data change + lens chnage: 0.991      0.977     NO CHANGE BASELINE: 0.986, 0.966
    # model = YOLO(os.path.join(this_dir, "yolov10x.pt"))
    # model = YOLO(os.path.join(this_dir, "yolov8s.pt"))
    results = model.train(
        data=os.path.join(this_dir, "/home/ecarlson/Downloads/multi-instance-object-detection-challenge/Starter_Dataset/yolo_params.yaml"), 
        epochs=args.epochs,
        device=0,
        single_cls=args.single_cls, 
        mosaic=args.mosaic,
        optimizer=args.optimizer, 
        lr0 = args.lr0, 
        lrf = args.lrf, 
        momentum=args.momentum,
        batch=8,
        close_mosaic=EPOCHS //2
    )
'''
Mixup boost val pred but reduces test pred
Mosaic shouldn't be 1.0  
'''


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''