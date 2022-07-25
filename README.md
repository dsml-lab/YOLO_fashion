# Fashion-YOLO
[YOLOv5](https://github.com/ultralytics/yolov5)に[Fashionpedia](https://fashionpedia.github.io/home/index.html)というデータセットを学習させて服装を検出できるようにしました．

## セットアップ
### YOLOv5のレポジトリをクローン
```
git clone https://github.com/ultralytics/yolov5.git
```
### コンテナを起動
```
docker-compose up -d
```
```
docker-compose exec fashion bash
```
### データセット用意
[ここ](https://github.com/cvdfoundation/fashionpedia#images)から画像をダウンロードします．
```
cd datasets/images
wget https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip
unzip train2020.zip
wget https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip
unzip val_test2020.zip
```
`setup_dataset.py`を実行して，データセットのアノテーションをダウンロードし，YOLOv5学習用に整形します．
```
python setup_dataset.py
```
## 学習
### 参考
- https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#3-train
- https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results
```
python yolov5/train.py --img 1024 --batch 128 --epochs 300 --data datasets/dataset.yaml --weights yolov5s.pt --device 0,1 --project fashion --name YOLOv5
```
## 検出可能なもの（~~46~~ 14カテゴリ）
0. shirt, blouse
0. top, t-shirt, sweatshirt
0. sweater
0. cardigan
0. jacket
0. vest
0. coat
0. cape
0. glasses
0. hat
0. headband, head covering, hair accessory
0. tie
0. watch
0. scarf

## 精度
![混同行列](src/confusion_matrix.png "混同行列")
