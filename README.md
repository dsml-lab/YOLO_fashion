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
0. shirt, blouse
0. top, t-shirt, sweatshirt
0. sweater
0. cardigan
0. jacket
0. vest
0. pants
0. shorts
0. skirt
0. coat
0. dress
0. jumpsuit
0. cape
0. glasses
0. hat
0. headband, head covering, hair accessory
0. tie
0. glove
0. watch
0. belt
0. leg warmer
0. tights, stockings
0. sock
0. shoe
0. bag, wallet
0. scarf
0. umbrella
0. hood
0. collar
0. lapel
0. epaulette
0. sleeve
0. pocket
0. neckline
0. buckle
0. zipper
0. applique
0. bead
0. bow
0. flower
0. fringe
0. ribbon
0. rivet
0. ruffle
0. sequin
0. tassel
