import os
import json
import yaml
import requests
from tqdm import tqdm


def create_dataset_yaml(data, filename):
    """
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#11-create-datasetyaml
    """
    names = [attr['name'] for attr in data['categories']]
    nc = len(names)
    data_yaml = {
        'path': '../../datasets',
        'train': 'images/train',
        'val': 'images/test',
        'nc': nc,
        'names': names,
    }
    with open(filename, 'w') as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)


def create_labels(data, label_dir):
    """
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#12-create-labels-1

    label_dir にはすでに存在するディレクトリを指定しない
    """
    os.makedirs(label_dir)
    images_info = {}
    # 画像情報をidから呼び出せるようにする
    for image_data in data['images']:
        images_info[image_data['id']] = image_data
    
    # アノテーション部分を取得
    for annotation in tqdm(data['annotations']):
        class_id = annotation['category_id']
        x, y, w, h = annotation['bbox']
        # bbox の中心の座標
        cx = (x + w) / 2
        cy = (y + h) / 2
        
        # アノテーションに対応する画像データを取得する
        image = images_info[annotation['image_id']]
        # 値を正規化する
        cx /= image['width']
        cy /= image['height']
        w /= image['width']
        h /= image['height']

        txt_filename = image['file_name'].rstrip('jpg') + 'txt'
        with open(os.path.join(label_dir, txt_filename), 'a') as txt_file:
            # class x_center y_center width height
            txt_file.write(f'{class_id} {cx} {cy} {w} {h}\n')


if __name__ == '__main__':
    train_path = 'datasets/instances_attributes_train2020.json'
    val_path = 'datasets/instances_attributes_val2020.json'

    print('Downloading train json file')
    train_data = requests.get('https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json').json()
    import pdb; pdb.set_trace()
    # print('Creating dataset yaml file.')
    # create_dataset_yaml(train_data, 'datasets/dataset.yaml')
    # print('Creating train labels.')
    # create_labels(train_data, 'datasets/labels/train')

    # print('Downloading val json file')
    # val_data = requests.get('https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json').json()
    # print('Creating val labels.')
    # create_labels(val_data, 'datasets/labels/test')
