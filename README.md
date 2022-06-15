# Fashion-YOLO
[YOLOv5](https://github.com/ultralytics/yolov5)に[Fashionpedia](https://fashionpedia.github.io/home/index.html)というデータセットを学習させて服装を検出できるようにしました．

## セットアップ
### YOLOv5のレポジトリをクローン
```
git clone https://github.com/ultralytics/yolov5.git
```
### condaの仮想環境構築
```
conda create -n fashion python=3.8
```
```
conda activate fashion
pip install yolov5/requirements.txt
```

## 検出デモ
webカメラから動画を取得してファッションアイテムを検出します．（GPU不要）
### 全画面表示
yolov5をそのまま使用すると，全画面表示できません．修正する場合は，`yolov5/detect.py`の176付近を変更してください．
### 実行
```
python yolov5/detect.py \
    --weights <学習後の重み（.ptファイル）> \
    --data fashionpedia.yaml \
    --source 0 \
    --device cpu \
    --view-img
```
- source: カメラのid.うまく行かない場合は，1や２に変えてみてください．
- device: GPUで実行する場合は， 0 または 0,1 のようにしてください．

終了するときには，ctrl+C で止めてください．

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
