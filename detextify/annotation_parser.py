import os
from typing import Dict, Optional, Sequence
from detextify.text_detector import TextBox


def convert_to_text_box(result: Dict) -> Optional[TextBox]:
    if 'original_width' not in result or 'original_height' not in result:
        return None
    
    value = result['value']
    # Since we made the annotations, we flipped the meaning of "height" and "weight" in TextBox to be more intuitive.
    # Now, height is for vertical and width is for horizontal.
    original_h = result["original_width"]   # notice meaning flip
    original_w = result["original_height"]  # notice meaning flip

    if all([key in value for key in ['x', 'y', 'height', 'width']]):
        x = value["x"]
        y = value["y"]
        h = value["width"]   # notice meaning flip
        w = value["height"]  # notice meaning flip

        return TextBox(x=int(x * original_h / 100.0),
                       y=int(y * original_w / 100.0),
                       h=int(h * original_h / 100.0),
                       w=int(w * original_w / 100.0))


def convert_to_text_boxes(annotation: Dict) -> Sequence[TextBox]:
    assert len(annotation["annotations"]) == 1
    gold = annotation["annotations"][0]
    text_boxes = [convert_to_text_box(result) for result in gold["result"]]
    return [tb for tb in text_boxes if tb is not None]


def get_image_basename(annotation: Dict) -> str:
    return os.path.basename("".join(annotation["data"]["data"].split("-")[1:]))
