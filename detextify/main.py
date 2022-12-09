"""Runs text detection over the folders in an image. Draws red rectangles around the detected text.

TODO: Remove this file from the library before publishing.
"""
from absl import app
from absl import flags
from basic_text_detector import BasicTextDetector

import cv2
import glob
import os
import utils

flags.DEFINE_string("input_dir", None, "Directory with input images.")
flags.DEFINE_string("output_dir", None, "Directory with output images, with text boxes drawn on the input image.")
flags.DEFINE_enum("mode", "BASIC", ["BASIC"], "Type of text detection.")
flags.mark_flags_as_required(["input_dir", "output_dir"])
FLAGS = flags.FLAGS


def main(_):
    if FLAGS.mode == "BASIC":
        detector = BasicTextDetector()
    else:
        raise NotImplemented(f"Mode `{FLAGS.mode}` for text detection is not implemented.")

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    for image_path in glob.glob(os.path.join(FLAGS.input_dir, "*")):
        text_boxes = detector.detect_text(image_path)
        name, ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(FLAGS.output_dir, name + "_boxes" + ext)
        utils.draw_text_boxes(text_boxes, image_path, output_path)


if __name__ == "__main__":
    app.run(main)
