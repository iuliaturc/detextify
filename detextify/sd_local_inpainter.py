from detextify.inpainter import Inpainter, make_black_and_white_mask
from detextify.text_detector import TextBox
from diffusers import StableDiffusionInpaintPipeline
from typing import Sequence
from PIL import Image

import tempfile
import torch

# This class is separate from `inpainter.py` because it depends on the torch library, which has many wheels.
# We'll ask users to manually install the right version for their system. However, for those users who don't want to
# use DiffusersSDInpainter, we want `from detextify import inpainter` to work (i.e., not fail on `import torch`).


class DiffusersSDInpainter(Inpainter):
  """Uses a Stable Diffusion model from HuggingFace for in-painting."""

  def __init__(self, pipe: StableDiffusionInpaintPipeline = None):
    if pipe is None:
      if not torch.cuda.is_available():
        raise Exception("You need a GPU + CUDA to run this model locally.")

      self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
          "stabilityai/stable-diffusion-2-inpainting",
          revision="fp16",
          torch_dtype=torch.float16).to("cuda")
    else:
      self.pipe = pipe

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], out_image_path: str):
    image = Image.open(in_image_path)

    mask_temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg")
    make_black_and_white_mask(text_boxes, image.height, image.width, mask_temp_file.name)
    mask_image = Image.open(mask_temp_file.name)

    out_image = self.pipe(prompt=Inpainter.PROMPT, image=image, mask_image=mask_image).images[0]
    out_image.save(out_image_path)
