# 画像データセット
[ここ](https://github.com/cvdfoundation/fashionpedia#annotations)から
- train2020.zip
- val_test2020.zip

をダウンロードして解凍してください．

```
.
|-- test
|   |-- 0a4aae5ecd970a120bfcc6b377b6e187.jpg
|   |-- 0a4f8205a3b58e70eec99fbbb9422d08.jpg
|   | 
|
`-- train
    |-- 0a0a539316af6547b3bbe228ead13730.jpg
    |-- 0a0f64ffdb6aa45b0f445b217b05a6c6.jpg
    |

```

`setup_dataset.py`を実行すると，testディレクトリ内の画像がtestとvalに分割されます．