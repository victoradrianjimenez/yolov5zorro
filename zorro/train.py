"""
Train a YOLOv5 model on a custom dataset using Zorro activation function.

Usage - Single-GPU training:
    $ python -m zorro.train --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python -m zorro.train --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3
"""

import sys
import runpy
sys.modules['models.yolo'] = sys.modules['zorro.yolo']

if __name__ == "__main__":
    runpy.__run_module_as_main('train')
