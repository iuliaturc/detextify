"""Utility methods."""
from text_detector import TextBox
from typing import Sequence

import cv2
import numpy as np


def draw_text_box(tb: TextBox, image: np.ndarray):
    """Draws a red rectangle around the text box. Modifies the array in place."""
    cv2.rectangle(image, (tb.x, tb.y), (tb.x + tb.h, tb.y + tb.w), (0, 0, 255), 2)


def draw_text_boxes(tbs: Sequence[TextBox], in_path: str, out_path: str):
    """Draws red rectangles around the given text boxes."""
    image = cv2.imread(in_path)
    for tb in tbs:
        draw_text_box(tb, image)
    cv2.imwrite(out_path, image)
