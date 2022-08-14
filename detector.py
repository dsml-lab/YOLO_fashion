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
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device


# DEVICE = 0
DEVICE = 'cpu'
WEIGHTS = ROOT / 'fashionpedia.pt'
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
        im0 = letterbox(image, self.imgsz, stride=self.model.stride, auto=self.model.pt)[0]
        # Convert
        img = im0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # Tensorにして正規化
        img = torch.from_numpy(img).float()
        img /= 255
        img = img[None]  # expand for batch dim

        img = img.to(self.device)
        pred = self.model(img)
        pred = non_max_suppression(pred)[0]

        # カテゴリごとの確信度を保持する
        coef_scores = [0. for _ in range(self.num_category)]
        # 予測結果を表示させるためのインスタンス
        annotator = Annotator(im0, line_width=3)
        # バウンディングボックスのスケールを揃える
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
        # カテゴリごとにクロップした画像を保存するため
        imc = im0.copy()

        import datetime
        timestamp = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")
        for *xyxy, conf, cls in reversed(pred):
            category_id = int(cls)
            # 確信度
            coef_scores[category_id] = max(coef_scores[category_id], conf.item())

            # 予測結果
            label = f'{self.category_names[category_id]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(category_id, True))
            # カテゴリごとにクロップした画像を保存する
            cropped_dir = Path(f'detect/categories/{self.category_names[category_id]}')
            os.makedirs(cropped_dir, exist_ok=True)
            save_one_box(xyxy, imc, file=cropped_dir / f'{timestamp}-{conf*100:.0f}.jpg', BGR=True)

        
        # 検出結果を保存する
        im0 = annotator.result()
        os.makedirs('detect/server/', exist_ok=True)
        save_path = f'detect/server/{timestamp}.jpg'
        import cv2
        cv2.imwrite(save_path, im0)
        
        response = dict(zip(self.category_names, coef_scores))

        return response


if __name__ == '__main__':
    import cv2
    img = cv2.imread('datasets/images/test/0cb2351c74ccc0b38128e91d0629625d.jpg')
    detector = FashionDetector()
    print(detector.detect(img))
