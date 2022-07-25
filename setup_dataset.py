import os
import requests
import shutil
from tqdm import tqdm
import yaml


def create_dataset_yaml(data, target_list, filename='datasets/dataset.yaml'):
    """dataset.yamlを作成する
    Args:
        data(dict): アノテーションのjsonファイルを辞書形式にしたもの
        target_list(list): 対象のカテゴリのindexを記録したリスト
        filename(str): yamlファイルの名前
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#11-create-datasetyaml
    """
    names = [attr['name'] for attr in data['categories'] if attr['id'] in target_list]
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


def create_labels(data, target_list, mode='train'):
    """アノテーションをyoloのフォーマットに変換する

    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#12-create-labels-1
    Args:
        data(dict): アノテーションのjsonファイルを辞書形式にしたもの
        target_list(list): 対象のカテゴリのindexを記録したリスト
        mode(str): ラベル付のモード（'train' or 'val'）
        dirname(str): アノテーションを保存するディレクトリ
    """
    if mode not in ['train', 'val']:
        ValueError(f"modeは'train','val'いずれかを指定してください mode={mode}")

    # ラベル用のディレクトリを作成
    label_dir = f'datasets/labels/{mode}'
    os.makedirs(label_dir)

    # 画像情報をidから呼び出せるようにする
    images_info = {}
    for image_data in data['images']:
        images_info[image_data['id']] = image_data

    # アノテーション部分を取得
    for annotation in tqdm(data['annotations']):
        if annotation['category_id'] in target_list:
            class_id = target_list.index(annotation['category_id'])
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


def split_val_test_img(data):
    """アノテーションがついているデータ(val)とついていないデータ(test)を分ける"""
    os.makedirs('datasets/images/val')
    # アノテーションがついている画像データを移動する
    for image_data in data['images']:
            shutil.move(
                f'datasets/images/test/{image_data["file_name"]}',
                'datasets/images/val/')


if __name__ == '__main__':
    # 検出対象のカテゴリID
    target_categories = [0, 1, 2, 3, 4, 5, 9, 12, 13, 14, 15, 16, 18, 25]
    yaml_filename = 'datasets/fashionpedia14.yaml'

    print('Downloading train json file')
    train_data = requests.get('https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json').json()
    print('Creating dataset yaml file.')
    create_dataset_yaml(train_data, target_categories, yaml_filename)
    print('Creating train labels.')
    create_labels(train_data, target_categories, mode='train')

    print('Downloading val json file')
    val_data = requests.get('https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json').json()
    print('Creating val labels.')
    create_labels(val_data, target_categories, mode='val')
