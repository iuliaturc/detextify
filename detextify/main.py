"""Runs text detection over the folders in an image. Draws red rectangles around the detected text.

TODO: Remove this file from the library before publishing.
"""
from absl import app
from absl import logging
from absl import flags
from azure_text_detector import AzureTextDetector
from basic_text_detector import BasicTextDetector

import annotation_parser
import cv2
import json
import glob
import numpy as np
import os
import utils

flags.DEFINE_string("input_dir", None, "Directory with input images.")
flags.DEFINE_string("output_dir", None, "Directory with output images, with text boxes drawn on the input image.")
flags.DEFINE_enum("mode", "BASIC", ["BASIC", "AZURE"], "Type of text detection.")
flags.DEFINE_string("azure_endpoint", None,
                    "Needed when --mode=AZURE. Should have the form https://<name>.cognitiveservices.azure.com/")
flags.DEFINE_string("azure_key", None, "Needed when --mode=AZURE.")
flags.DEFINE_string("annotations_file", None,
                    "Optional path to file with ground labels. If specified, we calculate IOU against text detection.")
flags.mark_flags_as_required(["input_dir", "output_dir"])
FLAGS = flags.FLAGS


def main(_):
    if FLAGS.mode == "BASIC":
        detector = BasicTextDetector()
    elif FLAGS.mode == "AZURE":
        if not FLAGS.azure_endpoint:
            raise ValueError("Please specify --azure_endpoint")
        if not FLAGS.azure_key:
            raise ValueError("Please specify --azure_key")
        detector = AzureTextDetector(FLAGS.azure_endpoint, FLAGS.azure_key)
    else:
        raise NotImplemented(f"Mode `{FLAGS.mode}` for text detection is not implemented.")

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    if not FLAGS.annotations_file:
        # Simply draw the detected text boxes.
        for image_path in glob.glob(os.path.join(FLAGS.input_dir, "*")):
            text_boxes = detector.detect_text(image_path)
            name, ext = os.path.splitext(os.path.basename(image_path))
            output_path = os.path.join(FLAGS.output_dir, name + f"_boxes_{FLAGS.mode}" + ext)
            utils.draw_text_boxes(text_boxes, image_path, output_path)
        return

    # Evaluate detected text boxes against gold annotations.
    ious = []
    with open(FLAGS.annotations_file) as f:
        annotations = json.load(f)
        for annotation in annotations:
            image_basename = annotation_parser.get_image_basename(annotation)
            image_path = os.path.join(FLAGS.input_dir, image_basename)
            if not os.path.exists(image_path):
                logging.error(f"Unable to find image {image_path}.")
                continue

            detected_boxes = detector.detect_text(image_path)
            # Text detection might produce granular boxes (e.g. one per word); merge them to resemble the annotations.
            merged_detected_boxes = utils.merge_nearby_boxes(detected_boxes, max_distance=30)
            golden_boxes = annotation_parser.convert_to_text_boxes(annotation)

            # For debugging purposes, draw all these boxes.
            image = cv2.imread(image_path)  # (B, G, R)
            for box in detected_boxes:
                utils.draw_text_box(box, image, color=(0, 255, 0), size=5)
            for box in merged_detected_boxes:
                utils.draw_text_box(box, image, color=(255, 0, 0), size=2)
            for box in golden_boxes:
                utils.draw_text_box(box, image, color=(0, 0, 255), size=2)
            output_path = os.path.join(FLAGS.output_dir, image_basename)
            cv2.imwrite(output_path, image)

            iou = utils.multi_intersection_over_union(merged_detected_boxes, golden_boxes)
            print(f"IOU = {iou} for {image_basename}")
            ious.append(iou)

    print("Average IOU across %d images: %f" % (len(ious), np.mean(ious)))


if __name__ == "__main__":
    app.run(main)
