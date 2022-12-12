Library to remove unwanted pseudo-text from images generated by AI
===
To run the main script (which will eventually be removed from the library):
poetry run python detextify/main.py

## Installing MMOCR
- Make sure you have pytorch, torchvision installed
```
poetry add openmim
mim install mmcv-full
pip install mmdet
pip install mmocr
# To install a given model's ckpt + config
mim download mmocr --config  textsnake_r50_fpn_unet_1200e_ctw1500 --dest .
```
