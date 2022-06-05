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
## 検出可能なもの（46カテゴリ）
1. shirt, blouse
1. top, t-shirt, sweatshirt
1. sweater
1. cardigan
1. jacket
1. vest
1. pants
1. shorts
1. skirt
1. coat
1. dress
1. jumpsuit
1. cape
1. glasses
1. hat
1. headband, head covering, hair accessory
1. tie
1. glove
1. watch
1. belt
1. leg warmer
1. tights, stockings
1. sock
1. shoe
1. bag, wallet
1. scarf
1. umbrella
1. hood
1. collar
1. lapel
1. epaulette
1. sleeve
1. pocket
1. neckline
1. buckle
1. zipper
1. applique
1. bead
1. bow
1. flower
1. fringe
1. ribbon
1. rivet
1. ruffle
1. sequin
1. tassel
