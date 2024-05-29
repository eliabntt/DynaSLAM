import os
import sys
import random
import math
import numpy as np

import torch, detectron2
import sys, os, distutils.core

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine.hooks import HookBase, BestCheckpointer
from detectron2.config import CfgNode
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager

cfg = get_cfg()

import json
from detectron2.data.datasets import register_coco_instances


def custom_config(num_classes, outdir, train, val, weights=''):
    cfg = get_cfg()

    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml"))
    cfg.DATASETS.TRAIN = (train,)
    cfg.DATASETS.TEST = (val,)
    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.MODEL.WEIGHTS = weights

    img_ratio = 4 / (8 * 2)
    cfg.SOLVER.IMS_PER_BATCH = 16 * img_ratio
    cfg.SOLVER.BASE_LR = 0.02 * img_ratio

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes

    cfg.OUTPUT_DIR = outdir

    cfg.SOLVER.BEST_CHECKPOINTER_SEGM = CfgNode({"ENABLED": True})
    cfg.SOLVER.BEST_CHECKPOINTER_SEGM.METRIC = "segm/AP50"
    cfg.SOLVER.BEST_CHECKPOINTER_SEGM.MODE = "max"

    cfg.SOLVER.BEST_CHECKPOINTER_BBOX = CfgNode({"ENABLED": True})
    cfg.SOLVER.BEST_CHECKPOINTER_BBOX.METRIC = "bbox/AP50"
    cfg.SOLVER.BEST_CHECKPOINTER_BBOX.MODE = "max"

    cfg.TEST.EVAL_PERIOD = 2000

    cfg.SOLVER.MAX_ITER = 270000 # --- [2 images] 120 --- [4 images] 90
    cfg.SOLVER.STEPS = (210000, 250000) # --- 80,108 --- 60K, 80K

    return cfg

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_subfolder="./eval"):
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, output_subfolder))
        evaluator_list = [coco_evaluator]
        return evaluator_list

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))

        if self.cfg.SOLVER.BEST_CHECKPOINTER_SEGM and comm.is_main_process():
            hooks.append(BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD,
                self.checkpointer,
                self.cfg.SOLVER.BEST_CHECKPOINTER_SEGM.METRIC,
                mode=self.cfg.SOLVER.BEST_CHECKPOINTER_SEGM.MODE,
                file_prefix="best_segm",
            ))

        if self.cfg.SOLVER.BEST_CHECKPOINTER_BBOX and comm.is_main_process():
            hooks.append(BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD,
                self.checkpointer,
                self.cfg.SOLVER.BEST_CHECKPOINTER_BBOX.METRIC,
                mode=self.cfg.SOLVER.BEST_CHECKPOINTER_BBOX.MODE,
                file_prefix="best_bbox",
            ))

        # swap the order of PeriodicWriter and ValidationLoss
        # code hangs with no GPUs > 1 if this line is removed
        hooks = hooks[:-2] + hooks[-2:][::-1]
        return hooks

    def build_writers(self):
        """
        Overwrites the default writers to contain our custom tensorboard writer

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        PathManager.mkdirs(self.cfg.OUTPUT_DIR)
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            CustomTensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

class Mask:
    """
    """
    def __init__(self): 
        print('Initializing Mask RCNN network...')
        # Root directory of the project
        ROOT_DIR = os.getcwd()
        ROOT_DIR = "./src/python"
        print(ROOT_DIR)

        
        weights_path = os.path.join(ROOT_DIR,"detectron2_models/base_model_detectron_r50_fpn.pkl")
        cfg = custom_config(80, "dummy","train_dummy", "val_dummy", weights_path)
        from detectron2.data import build_detection_test_loader
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST=0.3
        self.predictor = DefaultPredictor(cfg)
        
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
        print('Initialated2222 Mask RCNN network...')

    def GetDynSeg(self,image,image2=None):
        h = image.shape[0]
        w = image.shape[1]
        if len(image.shape) == 2:
            im = np.zeros((h,w,3))
            im[:,:,0]=image
            im[:,:,1]=image
            im[:,:,2]=image
            image = im
        #if image2 is not None:
        #	args+=[image2]
        # Run detection
        results = self.predictor(image)
        # Visualize results
        r = results["instances"].to("cpu").get_fields()
        i = 0
        mask = np.zeros((h,w))
        for roi in r['pred_classes']:
            if self.class_names[r['pred_classes'][i].item()] == 'person':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'bicycle':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'car':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'motorcycle':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'airplane':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'bus':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'train':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'truck':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'boat':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'bird':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'cat':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'dog':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'horse':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'sheep':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'cow':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'elephant':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'bear':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'zebra':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.
            if self.class_names[r['pred_classes'][i].item()] == 'giraffe':
                image_m = r['pred_masks'][i,:,:].numpy()
                mask[image_m == 1] = 1.		
            i+=1
#        print('GetSeg mask shape:',mask.shape)
#       print('GetSeg mask shape:',type(mask))

        return mask
