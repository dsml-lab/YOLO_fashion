import cv2
from PIL import Image
from io import BytesIO
import numpy as np
from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn

from detector import FashionDetector


app = FastAPI()
# 物体検出モデル
detector = FashionDetector()


@app.get("/")
def read_root():
    return {"Hello": "World"}


def read_image(image_encoded):
    # エンコードされた画像を読み込む
    img = Image.open(BytesIO(image_encoded))
    # numpy array に変換
    img = np.array(img)
    # RGBをBGRに変換
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


@app.post('/api/detect-fashion-items')
async def detect(file: UploadFile = File(...)):
    # 拡張子を確認
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    # 受け取ったファイルをBGR画像
    img = read_image(await file.read())
    return detector.detect(img)

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')
