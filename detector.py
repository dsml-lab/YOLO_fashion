import os
import sys
from pathlib import Path
import yaml

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
WEIGHTS = ROOT / 'fashion/fashionpedia14/weights/best.pt'
DATA_YAML = ROOT / 'datasets/dataset.yaml'
IMG_SIZE = (1024, 1024)


class FashionDetector:
    """ファッションアイテムを検出するクラス"""
    def __init__(self):
        # Load model
        self.device = select_device(DEVICE)
        self.model = DetectMultiBackend(WEIGHTS, device=self.device, data=DATA_YAML)

        self.imgsz = check_img_size(IMG_SIZE, s=self.model.stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

        self.set_categories(DATA_YAML)
    
    def set_categories(self, data_yaml):
        """データセットについて記述したyamlファイルからカテゴリを読み込む
        Args:
            data_yaml(str): データセットについて記述したyamlファイルのパス
        """
        with open(data_yaml) as f:
            data_config = yaml.safe_load(f)
        self.num_category = data_config['nc']
        self.category_names = data_config['names']
    
    def detect(self, image):
        """ファッションアイテムを検出する
        Args:
            image(np.ndarray): 検出対象の画像
        Returns:
            response(dict): 各カテゴリの確信度（２つ以上検出した場合は，最大値）
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

        # カテゴリごとの確信度を保持する
        coef_scores = [0. for _ in range(self.num_category)]
        for pred in preds:
            conf = pred[-2].cpu().item()
            category_id = int(pred[-1])
            coef_scores[category_id] = max(coef_scores[category_id], conf)
        
        response = dict(zip(self.category_names, coef_scores))

        return response


if __name__ == '__main__':
    import cv2
    img = cv2.imread('datasets/images/test/0cb2351c74ccc0b38128e91d0629625d.jpg')
    detector = FashionDetector()
    print(detector.detect(img))
