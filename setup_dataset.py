import os
import requests
import shutil
from tqdm import tqdm
import yaml


def create_dataset_yaml(data):
    """dataset.yamlを作成する
    Args:
        data(dict): アノテーションのjsonファイルを辞書形式にしたもの

    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#11-create-datasetyaml
    """
    filename = 'datasets/dataset.yaml'
    names = [attr['name'] for attr in data['categories']]
    nc = len(names)
    data_yaml = {
        'path': '../datasets',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': nc,
        'names': names,
    }
    with open(filename, 'w') as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)


def create_labels(data, mode='train'):
    """アノテーションをyoloのフォーマットに変換する

    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#12-create-labels-1
    Args:
        data(dict): アノテーションのjsonファイルを辞書形式にしたもの
        mode(str): ラベル付のモード（'train' of 'val'）
    """
    if mode not in ['train', 'val']:
        ValueError(f"modeは'train','val'いずれかを指定してください mode={mode}")

    # ラベル用のディレクトリを作成
    label_dir = f'datasets/labels/{mode}'
    os.makedirs(label_dir)

    # val用のディレクトリを作成する
    if mode == 'val':
        os.makedirs('datasets/images/val')

    # 画像情報をidから呼び出せるようにする
    images_info = {}
    for image_data in data['images']:
        images_info[image_data['id']] = image_data
        # アノテーションがついている画像データを移動する
        if mode == 'val':
            shutil.move(
                f'datasets/images/test/{image_data["file_name"]}',
                'datasets/images/val/')

    # アノテーション部分を取得
    for annotation in tqdm(data['annotations']):
        class_id = annotation['category_id']
        x, y, w, h = annotation['bbox']
        # bbox の中心の座標
        cx = x + w/2
        cy = y + h/2

        # アノテーションに対応する画像データを取得する
        image = images_info[annotation['image_id']]
        # 値を正規化する
        cx /= image['width']
        cy /= image['height']
        w /= image['width']
        h /= image['height']

        txt_filename = image['file_name'].replace('.jpg','.txt')
        with open(os.path.join(label_dir, txt_filename), 'a') as txt_file:
            # class x_center y_center width height
            txt_file.write(f'{class_id} {cx} {cy} {w} {h}\n')


if __name__ == '__main__':

    print('Downloading train json file')
    train_data = requests.get('https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json').json()
    print('Creating dataset yaml file.')
    create_dataset_yaml(train_data)
    print('Creating train labels.')
    create_labels(train_data)

    print('Downloading val json file')
    val_data = requests.get('https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json').json()
    print('Creating val labels.')
    create_labels(val_data, mode='val')
