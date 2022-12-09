"""Basic text detector based on cv2. Only works when the image has text and no other objects."""
from text_detector import TextBox, TextDetector
from typing import Sequence

import cv2
import os


class BasicTextDetector(TextDetector):
    """Basic text detector based on cv2. Only works when the image has text and no other objects."""

    def __init__(self, debug_folder: str = None):
        self.debug_folder = debug_folder
        if debug_folder and not os.path.exists(debug_folder):
            os.makedirs(debug_folder)

    def detect_text(self, image_path: str) -> Sequence[TextBox]:
        image = cv2.imread(image_path)
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(grayed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # A smaller value like (10, 10) will detect each word instead of a sentence.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(binary, rect_kernel, iterations=1)

        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        text_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            text_boxes.append(TextBox(x, y, w, h))

        if self.debug_folder:
            # For debugging, output intermediate stages of the pipeline.
            name, ext = os.path.splitext(os.path.basename(image_path))
            cv2.imwrite(os.path.join(self.debug_folder, name + "_grayed" + ext), grayed)
            cv2.imwrite(os.path.join(self.debug_folder, name + "_binary" + ext), binary)
            cv2.imwrite(os.path.join(self.debug_folder, name + "_dilation" + ext), dilation)

        return text_boxes
