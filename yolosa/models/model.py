import numpy as np
from PIL.Image import Image
from pathlib import Path
import torch
from torch import nn
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.nn.modules import C2f, Conv
from ultralytics.utils import yaml_load, ROOT
from yolosa.models.sam.utils.transforms import ResizeLongestSide
from yolosa.models.sam.build_sam import build_custom_sam
from yolosa.models.sam.predictor import SamPredictor
from ultralytics import YOLOWorld

class YOLOSAWorldSAImageEncoder(nn.Module):
    # pixel_mean = [123.675, 116.28, 103.53]
    # pixel_std = [58.395, 57.12, 57.375]
    img_size = 1024
    
    def __init__(self, backbone, adapter):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        # self.transform = ResizeLongestSide(self.img_size)
    
    def forward(self, x):
        output = []
        # if not isinstance(x, torch.Tensor):
        #     x = self.preprocess(x)
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            output.append(self.adapter[i](x))
        return sum(output)
        
    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        if isinstance(x, Image):
            x = np.array(Image)
        x = self.transform.apply_image(x)
        x = torch.as_tensor(x, device=self.device)
        x = x.permute(2, 0, 1).contiguous()[None, :, :, :]
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = torch.nn.functional.pad(x, (0, padw, 0, padh))
        return x

class YOLOSAWorld(YOLOWorld):
    """YOLO-World object detection model."""
    downscale = 16
    sam_backnone_out_dim = 256
    def __init__(self, model="yolov8s-world.pt", *args, **kwargs) -> None:
        """
        Initializes the YOLOv8-World model with the given pre-trained model file. Supports *.pt and *.yaml formats.

        Args:
            model (str): Path to the pre-trained model. Defaults to 'yolov8s-world.pt'.
        """
        super().__init__(model=model)

        # Assign default COCO class names
        self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")
        sam_backbone = self.set_sam_backbone()
        sam_adapter = self.set_sam_adapter(sam_backbone)
        self.set_sam()
        if 'sam_ckpt' in kwargs.keys():
            sam_ckpt = torch.load(kwargs['sam_ckpt'])
            sam_adapter.load_state_dict(sam_ckpt['adapter'])
            self.sam.load_state_dict(sam_ckpt['prompt_encoder_sam_decoder'])
        # if 'decoder_ckpt' in kwargs.keys():
            # self.set_sam(kwargs['decoder_ckpt'])
            # decoder_ckpt = torch.load(kwargs['decoder_ckpt'])
            # self.sam.load_state_dict(decoder_ckpt)
        self.sam.image_encoder = YOLOSAWorldSAImageEncoder(sam_backbone, sam_adapter)
        # else:
        #     sam_ckpt = None 
        # if 'adapter_ckpt' in kwargs.keys():
        #     self.sam_adapter.load_state_dict(torch.load(kwargs['adapter_ckpt']))
        

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }
    
    def set_sam_backbone(self):
        sam_backbone = nn.ModuleList()
        backbone = self.model.model[:9]
        sam_backbone.append(backbone[:2])
        sam_backbone.append(backbone[2:4])
        sam_backbone.append(backbone[4:6])
        return sam_backbone
    
    def set_sam_adapter(self, sam_backbone):
        sam_adapter = nn.ModuleList()
        cur_downscale = 4
        for stage in sam_backbone:
            stage_out_dim = stage[-1].conv.weight.data.shape[0]
            k = self.downscale // cur_downscale
            cur_downscale *= 2
            sam_adapter.append(nn.Sequential(nn.Conv2d(stage_out_dim, self.sam_backnone_out_dim, kernel_size = k, stride=k),
                                                  C2f(self.sam_backnone_out_dim, self.sam_backnone_out_dim)))
        return sam_adapter
    
    # def sam_backbone_forward(self, x):
    #     output = []
    #     if not isinstance(x, torch.Tensor):
    #         x = self.preprocess(x)
    #     for i in range(len(self.sam_backbone)):
    #         # x = self.sam_adapter[i](self.sam_backnone[i](x))
    #         x = self.sam_backbone[i](x)
    #         # print()
    #         output.append(self.sam_adapter[i](x))
    #     return sum(output)
        # return output
    
    def set_sam(self, sam_ckpt=None):
        self.sam = build_custom_sam(sam_ckpt)
        self.sam_predictor = SamPredictor(self.sam)
    
    # def preprocess(self, x):
    #     """Normalize pixel values and pad to a square input."""
    #     if isinstance(x, Image):
    #         x = np.array(Image)
    #     x = self.transform.apply_image(x)
    #     x = torch.as_tensor(x, device=self.device)
    #     x = x.permute(2, 0, 1).contiguous()[None, :, :, :]
    #     # Normalize colors
    #     x = (x - self.pixel_mean) / self.pixel_std

    #     # Pad
    #     h, w = x.shape[-2:]
    #     padh = self.img_size - h
    #     padw = self.img_size - w
    #     x = torch.nn.functional.pad(x, (0, padw, 0, padh))
    #     return x
