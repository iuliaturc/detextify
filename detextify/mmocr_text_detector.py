import os
from typing import Sequence

import cv2
import numpy as np
import requests
from mmocr.utils.ocr import MMOCR

from detextify.text_detector import TextBox
from detextify.text_detector import TextDetector

MODEL_CONFIG_DIR = "model_conf/textsnake_r50_fpn_unet_1200e_ctw1500.py"
MODEL_CKPT_URL = "https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth"
MODEL_CKPT_DIR = "model_conf/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth"


class MMOCRTextDetector(TextDetector):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        # TODO: Download model if not already present in `cache_dir`
        if not os.path.exists(MODEL_CKPT_DIR):
            print("Could not find local copy of text detection model. Downloading a copy to models_conf/")
            resp = requests.get(MODEL_CKPT_URL)
            with open(MODEL_CKPT_DIR, "wb") as f:
                f.write(resp.content)
        self.mmocr = MMOCR(det="TextSnake",
                           det_config=MODEL_CONFIG_DIR,
                           det_ckpt=MODEL_CKPT_DIR,
                           recog=None)
    
    def detect_text(self, image_path: str) -> Sequence[TextBox]:
        output = self.mmocr.readtext(image_path,
                                     output=self.output_dir)
        # Convert to bounding rectangle
        boundaries = output[0]["boundary_result"]
        new_boundaries = []
        for boundary in boundaries:
            poly = np.array(boundary[:-1]).reshape(-1, 2).astype(np.float32)
            # NOTE: This `h` and `w` don't match the TextBox definitions of `h` and `w`
            x, y, h, w = cv2.boundingRect(poly)
            new_boundaries.append(TextBox(x=x, y=y, h=h, w=w))
        return new_boundaries
