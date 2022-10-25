# Model-based Radial Distortion Rectification via Transformer


## Inference 
1. Download the pretrained models from [Baidu Cloud](), and put them to `$ROOT/model_pretrained/`.
2. Put the distorted images in `$ROOT/distorted/`.
3. Distortion rectification. The rectified images are saved in `$ROOT/result/` by default.
    ```
    python inference.py
    ```

## dataset
The codes of dataset genaration are based on [PCN](https://github.com/uof1745-cmd/PCN) and [Blind](https://github.com/xiaoyu258/GeoProj). Thanks for their wonderful works.
