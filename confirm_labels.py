import os
import cv2
import glob
import copy

image_dir = 'datasets/images/test'
label_dir = 'datasets/labels/test'
image_filenames = os.listdir(image_dir)
# import pdb; pdb.set_trace()
# image_paths = glob.glob(os.path.join(dataset_path, 'images', 'test', '*.jpg'))
# label_paths = glob.glob(os.path.join(dataset_path, 'labels', 'test', '*.txt'))
# image_paths = ['datasets/images/test/99601fa457d157b81154d089966c2e3a.jpg']
check_dir = 'check'
os.makedirs(check_dir, exist_ok=True)
# BountingBoxを描画
for image_filename in image_filenames:
    image_path = os.path.join(image_dir, image_filename)
    # 画像の読み込み
    im = cv2.imread(image_path)
    im_draw = im.copy()
    h, w, c = im.shape
    # bboxの情報を読み込み
    label_filename = image_filename.replace('.jpg', '.txt')
    label_path = os.path.join(label_dir, label_filename)
    try:
        with open(label_path, "r") as txt_file: 
            bbox_list = txt_file.readlines()
    except:
        print('check', image_filename)
        continue

    for bbox in bbox_list:
        bb_cx, bb_cy, bb_w, bb_h = bbox.replace('\n', '').split(' ')[1:]
        bb_cx, bb_cy, bb_w, bb_h = float(bb_cx)*w, float(bb_cy)*h, float(bb_w)*w, float(bb_h)*h
        tlx, tly, brx, bry = int(bb_cx-bb_w/2), int(bb_cy-bb_h/2), int(bb_cx+bb_w/2), int(bb_cy+bb_h/2)
        # 四角を描画(左上, 右下)
        im_draw = cv2.rectangle(im_draw, (tlx, tly), (brx, bry), (0, 0, 255), thickness=2)
        im_draw = cv2.circle(im_draw, (int(bb_cx), int(bb_cy)), 10, (0,0,255), thickness=-1)
    # 画像の書き込み
    cv2.imwrite(os.path.join(check_dir, image_filename), im_draw )
