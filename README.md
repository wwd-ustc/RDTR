# Model-aware Representation Learning for Radial Distortion Rectification

![1](https://user-images.githubusercontent.com/93323070/197723600-bf6a78d9-e17b-48f7-9a5d-2a0c136dd551.png)

## Inference 
1. For barrel distortion rectification, download the pretrained models from [Baidu Cloud](https://pan.baidu.com/s/1gZXrfw5Z_0Uz-X5FJtDT2g?pwd=76fx)(Extraction code: 76fx). For pincushion distortion rectification, download the pretrained models from [Baidu Cloud](https://pan.baidu.com/s/1OuN_DgKU30fzmEFj3pDMwg?pwd=fabg)(Extraction code: fabg). Put the model to `$ROOT/model_pretrained/`.
2. Put the distorted images in `$ROOT/distorted/`.
3. Distortion rectification. The rectified images are saved in `$ROOT/result/` by default.
    ```
    python inference.py
    ```

## Dataset
For training the network, you need to download the source data from [Place365 dataset](http://places2.csail.mit.edu/download.html).
The codes of dataset genaration are based on [PCN](https://github.com/uof1745-cmd/PCN) and [Blind](https://github.com/xiaoyu258/GeoProj). Thanks for their wonderful works.
