from detextify.inpainter import Inpainter
from detextify.text_detector import TextDetector


class Detextifier:
    def __init__(self, text_detector: TextDetector, inpainter: Inpainter):
        self.text_detector = text_detector
        self.inpainter = inpainter

    def detextify(self, in_image_path: str, out_image_path: str, prompt=Inpainter.DEFAULT_PROMPT, max_retries=5):
        to_inpaint_path = in_image_path
        for i in range(max_retries):
            print(f"Iteration {i} of {max_retries} for image {in_image_path}:")

            print(f"\tCalling text detector...")
            text_boxes = self.text_detector.detect_text(to_inpaint_path)
            print(f"\tDetected {len(text_boxes)} text boxes.")

            if not text_boxes:
                break

            print(f"\tCalling in-painting model...")
            self.inpainter.inpaint(to_inpaint_path, text_boxes, prompt, out_image_path)
            to_inpaint_path = out_image_path


from detextify.paddle_text_detector import PaddleTextDetector
from detextify.inpainter import ReplicateSDInpainter
from detextify.detextifier import Detextifier
from absl import app

def main(argv):
  td = PaddleTextDetector()
  inp = ReplicateSDInpainter("3480af9c7d0d8b1742c7c3f0fca8ef70d7820eb2")
  det = Detextifier(td, inp)
  det.detextify("../images/octopus.png", "../images/octopus_out.png")

if __name__ == "__main__":
    app.run(main)
