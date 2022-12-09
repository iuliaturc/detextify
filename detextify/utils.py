"""Utility methods."""
from text_detector import TextBox
from typing import Sequence

import cv2
import numpy as np


def draw_text_box(tb: TextBox, image: np.ndarray, color=(0, 0, 255)):
    """Draws a red rectangle around the text box. Modifies the array in place."""
    cv2.rectangle(image, (tb.x, tb.y), (tb.x + tb.h, tb.y + tb.w), color, 2)


def draw_text_boxes(tbs: Sequence[TextBox], in_path: str, out_path: str, color=(0, 0, 255)):
    """Draws red rectangles around the given text boxes."""
    image = cv2.imread(in_path)
    for tb in tbs:
        draw_text_box(tb, image, color)
    cv2.imwrite(out_path, image)


def intersection_over_union(box1: TextBox, box2: TextBox):
    # Determine the (x, y)-coordinates of the intersection rectangle.
    xa = max(box1.x, box2.x)
    ya = max(box1.y, box2.y)
    xb = min(box1.x + box1.h, box2.x + box2.h)
    yb = min(box1.y + box1.w, box2.y + box2.w)
    # Compute the area of intersection rectangle.
    intersection_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (box1.h + 1) * (box1.w + 1)
    box2_area = (box2.h + 1) * (box2.w + 1)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou
