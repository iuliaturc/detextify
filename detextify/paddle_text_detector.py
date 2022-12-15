from detextify.text_detector import TextBox, TextDetector
from paddleocr import PaddleOCR
from typing import Sequence
from PIL import Image

import logging
paddle_logger = logging.getLogger("ppocr").setLevel(logging.ERROR)

# This class is separate from `text_detector.py` because it depends on the paddle library, which has many wheels
# (see https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html).
# We'll ask users to manually install the right version for their system. However, for those users who don't want to
# use/install paddle, we want `from detextify import text_detector` to work (i.e., not fail on `import paddleocr`).


class PaddleTextDetector(TextDetector):
  """Uses PaddleOCR for text detection: https://github.com/PaddlePaddle/PaddleOCR"""

  def __init__(self, pad_size: int = 30):
    """
    Args:
      pad_size: Empirically, Paddle returns text boxes that are quite tight. When we mask out the text boxes and the
        text is not masked out completely, the in-painting models regenerate text. The padding adds a safety margin to
        the text boxes to prevent such situations.
    """
    self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang="en")
    self.pad_size = pad_size

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
        logger.error(f"Malformed bounding box from Paddle: {points}")

      if self.pad_size:
          tl_x = max(0, tl_x - self.pad_size)
          tl_y = max(0, tl_y - self.pad_size)

          image = Image.open(image_path)
          max_h = image.height - tl_x
          h = min(h + self.pad_size, image.height - tl_x)
          w = min(w + self.pad_size, image.width - tl_y)

      text_boxes.append(TextBox(int(tl_x), int(tl_y), int(h), int(w), text))
    return text_boxes

