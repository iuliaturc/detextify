from absl import logging
from inpainter import Inpainter
from text_detector import TextDetector


class Detextifier:
    def __init__(self, text_detector: TextDetector, inpainter: Inpainter):
        self.text_detector = text_detector
        self.inpainter = inpainter

    def detextify(self, in_image_path: str, out_image_path: str, max_retries=5):
        to_inpaint_path = in_image_path
        for i in range(max_retries):
            logging.info(f"Inpainting iteration {i} for image {in_image_path}...")
            text_boxes = self.text_detector.detect_text(to_inpaint_path)
            if not text_boxes:
                break
            self.inpainter.inpaint(to_inpaint_path, text_boxes, out_image_path)
            to_inpaint_path = out_image_path
