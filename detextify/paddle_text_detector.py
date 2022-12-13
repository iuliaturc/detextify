from paddleocr import PaddleOCR
from text_detector import TextBox, TextDetector
from typing import Sequence


class PaddleTextDetector(TextDetector):
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang="en")

    def detect_text(self, image_path: str) -> Sequence[TextBox]:
        result = self.ocr.ocr(image_path, cls=True)[0]
        text_boxes = []
        for line in result:
            points = line[0]
            text = line[1][0]
            # These points are not necessarily a rectangle, but rather a polygon.
            # We'll find the smallest enclosing rectangle.
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            tl_x = min(xs)
            tl_y = min(ys)
            h = max(xs) - tl_x
            w = max(ys) - tl_y

            if h < 0 or w < 0:
                logging.error(f"Malformed bounding box from Paddle: {points}")

            text_boxes.append(TextBox(int(tl_x), int(tl_y), int(h), int(w), text))
        return text_boxes
