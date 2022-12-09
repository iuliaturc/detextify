import argparse
import json
import os
from typing import Dict

import cv2


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-path", type=str)
    parser.add_argument("--images-base-dir", type=str)
    return parser.parse_args()


def convert_from_ls(result):
    if 'original_width' not in result or 'original_height' not in result:
        return None
    
    value = result['value']
    w, h = result['original_width'], result['original_height']
    
    if all([key in value for key in ['x', 'y', 'width', 'height']]):
        return w * value['x'] / 100.0, \
               h * value['y'] / 100.0, \
               w * value['width'] / 100.0, \
               h * value['height'] / 100.0


def draw_box(img_path: str, annotation: Dict):
    image = cv2.imread(img_path)
    window_name = "Image"
    # NOTE: This only draws a single bounding box even if there are multiple ones annotated
    bbox_num = 0
    x, y, width, height = convert_from_ls(annotation[0]["result"][bbox_num])
    
    start_point = (int(x), int(y))
    # represents the bottom right corner of rectangle
    end_point = (int(x + width), int(y + height))
    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    # Displaying the image
    cv2.imshow(window_name, image)
    cv2.waitKey()


if __name__ == "__main__":
    args = read_args()
    with open(args.annotation_path) as f:
        annotations = json.load(f)
    
    annotation_num = 16
    img_rel_path = os.path.basename("".join(annotations[annotation_num]["data"]["data"].split("-")[1:]))
    draw_box(os.path.join(args.images_base_dir, img_rel_path), annotations[annotation_num]["annotations"])
