"""Utility methods."""
from text_detector import TextBox
from typing import List, Sequence

import cv2
import itertools
import numpy as np


def draw_text_box(tb: TextBox, image: np.ndarray, color=(0, 0, 255), size=2):
    """Draws a red rectangle around the text box. Modifies the array in place."""
    cv2.rectangle(image, (tb.x, tb.y), (tb.x + tb.h, tb.y + tb.w), color, size)


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


def merge_nearby_boxes(boxes: Sequence[TextBox], max_distance) -> Sequence[TextBox]:
    """Merges boxes that are less than `max_distance` pixels apart on both the x and y axes."""
    if len(boxes) <= 1:
        return boxes

    def overlap_x(box1: TextBox, box2: TextBox) -> int:
        return min(box1.x + box1.h, box2.x + box2.h) - max(box1.x, box2.x)

    def overlap_y(box1: TextBox, box2: TextBox) -> int:
        return min(box1.y + box1.w, box2.y + box2.w) - max(box1.y, box2.y)

    def should_merge(box1: TextBox, box2: TextBox) -> bool:
        # Boxes need to overlap on one axis and be close to each other on the other axis.
        # Note that the inverse of overlap is distance.
        x_overlap = overlap_x(box1, box2)
        y_overlap = overlap_y(box1, box2)
        return (x_overlap > 0 and -y_overlap < max_distance) or (y_overlap > 0 and -x_overlap < max_distance)

    def merge(bs: Sequence[TextBox]) -> TextBox:
        """Merges boxes into the smallest enclosing rectangle."""
        tl = (min([b.x for b in bs]), min([b.y for b in bs]))
        br = (max([b.x + b.h for b in bs]), max([b.y + b.w for b in bs]))
        return TextBox(x=tl[0], y=tl[1], h=br[0] - tl[0], w=br[1] - tl[1])

    def merge_with_box(ref_box: TextBox, other_boxes: List[TextBox]) -> List[TextBox]:
        """Merges `ref_box` with boxes from `other_boxes` that are close enough. Returns the other boxes unchanged."""
        should_merge_with_ref = [should_merge(ref_box, box) for box in other_boxes]

        if sum(should_merge_with_ref) == 0:
            return [ref_box] + other_boxes

        to_merge = list(itertools.compress(other_boxes, should_merge_with_ref))
        merged_box = merge([ref_box] + to_merge)
        should_keep = [not should for should in should_merge_with_ref]
        to_keep = list(itertools.compress(other_boxes, should_keep))
        return [merged_box] + to_keep

    curr_boxes = boxes
    ref_idx = 0
    while ref_idx < len(curr_boxes) - 1:
        before = curr_boxes[:ref_idx]
        after = merge_with_box(curr_boxes[ref_idx], curr_boxes[ref_idx:])
        if len(before) + len(after) == len(curr_boxes):
            # No merge happened. We can advance the index.
            ref_idx += 1
        curr_boxes = before + after

    return curr_boxes
