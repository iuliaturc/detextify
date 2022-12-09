from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

import time
from absl import logging
from text_detector import TextBox, TextDetector
from typing import Sequence


class AzureTextDetector(TextDetector):
    """Calls the Computer Vision endpoint from Microsoft Azure. Promises to work with images in the wild."""

    def __init__(self, endpoint, key):
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    def detect_text(self, image_filename: str) -> Sequence[TextBox]:
        read_response = self.client.read_in_stream(open(image_filename, "rb"), raw=True)

        # Get the operation location (URL with an ID at the end) from the response
        read_operation_location = read_response.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for it to retrieve the results
        while True:
            read_result = self.client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        text_boxes = []
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    # line.bounding_box contains the 4 corners of a polygon (not necessarily a rectangle).
                    # To keep things simple, we turn them into rectangles. There are two ways: (1) use the rectangle
                    # defined by the top-left and bottom-right corners, or (2) use the rectangle that encompasses the
                    # entire polygon. (1) will lead to smaller surfaces, (2) to bigger surfaces.

                    # Implementation for (1)
                    # tl_x, tl_y = line.bounding_box[0:2]     # top left
                    # br_x, br_y = line.bounding_box[4:6]     # bottom right
                    # w = br_x - tl_x
                    # h = br_y - tl_y

                    # Implementation for (2)
                    xs = [point for idx, point in enumerate(line.bounding_box) if idx % 2 == 0]
                    ys = [point for idx, point in enumerate(line.bounding_box) if idx % 2 == 1]
                    tl_x = min(xs)
                    tl_y = min(ys)
                    w = max(xs) - tl_x
                    h = max(ys) - tl_y

                    if w < 0 or h < 0:
                        logging.error(f"Malformed bounding box from Azure: {line.bounding_box}")

                    text_boxes.append(TextBox(int(tl_x), int(tl_y), int(w), int(h), line.text))
        return text_boxes
