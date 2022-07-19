import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
sys.path.append('yolov5')
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import numpy as np
import torch

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression
from yolov5.utils.torch_utils import select_device


DEVICE = 0
WEIGHTS = ROOT / 'fashion/YOLOv5/weights/best.pt'
DATA_YAML = ROOT / 'datasets/dataset.yaml'
DEVICE = 0
IMG_SIZE = (1024, 1024)


class FashionDetector:
    """ファッションアイテムを検出するクラス

    Attributes:
        device: GPU
        model: YOLOのモデル
        stride:
        names:
        pt:
        imgsz: 画像のサイズ
    """
    def __init__(self):
        # Load model
        self.device = select_device(DEVICE)
        self.model = DetectMultiBackend(
            WEIGHTS, device=self.device, data=DATA_YAML)

        self.imgsz = check_img_size(IMG_SIZE, s=self.model.stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
    
    def detect(self, image):
        """ファッションアイテムを検出する
        Args:
            image(numpy.ndarray): 検出対象の画像
        Returns:
            numpy.ndarray: (46, )の各カテゴリの確信度（２つ以上検出した場合は，最大値）
        """
        # Padded resize
        img = letterbox(image, self.imgsz, stride=self.model.stride, auto=self.model.pt)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # Tensorにして正規化
        img = torch.from_numpy(img).float()
        img /= 255
        img = img[None]  # expand for batch dim

        img = img.to(self.device)
        pred = self.model(img)
        preds = non_max_suppression(pred)[0]

        # カテゴリごと（46カテゴリ）の確信度を保持する
        response = np.zeros(46)
        for pred in preds:
            conf = pred[-2].cpu().item()
            category_id = int(pred[-1])
            response[category_id] = max(response[category_id], conf)

        return response





import cv2
img = cv2.imread('datasets/images/test/0a45b6b033bf0077a15f484e98f3dbfe.jpg')
detector = FashionDetector()
print(detector.detect(img))