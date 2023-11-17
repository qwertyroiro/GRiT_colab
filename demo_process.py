import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

# Pythonの実行場所からの相対パスを記述しないとimportだめっぽい。
# だから、はじめに"GRiT_colab/"を付与した。
sys.path.insert(0, 'GRiT_colab/third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
sys.path.insert(0, 'GRiT_colab/')
from grit.config import add_grit_config

from grit.predictor import VisualizationDemo

# constants
WINDOW_NAME = "GRiT"

def setup_cfg(cpu=False, test_task=""):
    cfg = get_cfg()
    if cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file("GRiT_colab/configs/GRiT_B_DenseCap_ObjectDet.yaml")
    opts = ["MODEL.WEIGHTS", "GRiT_colab/models/grit_b_densecap_objectdet.pth"]
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    if test_task:
        cfg.MODEL.TEST_TASK = test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg

def GRiT_process(img):
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    cfg = setup_cfg()
    demo = VisualizationDemo(cfg)
    predictions, visualized_output = demo.run_on_image(img)
    
    # 中身をcpuに移動
    instances = predictions["instances"].to(torch.device("cpu"))
    # 検出したbox
    pred_boxes = instances.pred_boxes.tensor.detach().numpy().tolist()
    # それへのキャプショニング
    pred_object_descriptions = instances.pred_object_descriptions.data
    return pred_boxes, pred_object_descriptions
