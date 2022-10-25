# Model-aware Representation Learning for Radial Distortion Rectification

[page1-v4-crop.pdf](https://github.com/wwd-ustc/RDTR/files/9858508/page1-v4-crop.pdf)

## Inference 
1. Download the pretrained models from [Baidu Cloud](), and put them to `$ROOT/model_pretrained/`.
2. Put the distorted images in `$ROOT/distorted/`.
3. Distortion rectification. The rectified images are saved in `$ROOT/result/` by default.
    ```
    python inference.py
    ```

## Dataset
For training the network, you need to download the source data from [Place365 dataset](http://places2.csail.mit.edu/download.html).
The codes of dataset genaration are based on [PCN](https://github.com/uof1745-cmd/PCN) and [Blind](https://github.com/xiaoyu258/GeoProj). Thanks for their wonderful works.
