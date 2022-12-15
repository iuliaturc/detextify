from detextify.inpainter import Inpainter, make_black_and_white_mask
from detextify.text_detector import TextBox
from diffusers import StableDiffusionInpaintPipeline
from typing import Sequence
from PIL import Image

import detextify.utils as utils
import math
import tempfile
import torch

# This class is separate from `inpainter.py` because it depends on the torch library, which has many wheels.
# We'll ask users to manually install the right version for their system. However, for those users who don't want to
# use DiffusersSDInpainter, we want `from detextify import inpainter` to work (i.e., not fail on `import torch`).


def tile_has_text_box(crop_x: int, crop_y: int, crop_size: int, text_boxes: Sequence[TextBox]):
  # Turn the tile into a TextBox just so that we can reuse utils.boxes_intersect
  crop_box = TextBox(crop_x, crop_y, crop_size, crop_size)
  return any([utils.boxes_intersect(crop_box, text_box) for text_box in text_boxes])


def pad_to_size(image, size):
  new_image = Image.new(image.mode, (size, size), color=(0, 0, 0))
  new_image.paste(image)
  return new_image


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

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], prompt: str, out_image_path: str):
    image = Image.open(in_image_path)

    mask_temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg")
    make_black_and_white_mask(text_boxes, image.height, image.width, mask_temp_file.name)
    mask_image = Image.open(mask_temp_file.name)

    # SD only accepts images that are exactly 512 x 512.
    SD_SIZE = 512

    if image.height == SD_SIZE and image.width == SD_SIZE:
      out_image = self.pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    else:
      # Break the image into 512 x 512 tiles. In-paint the tiles that contain text boxes.
      out_image = image.copy()

      for x in range(0, image.height, SD_SIZE):
        for y in range(0, image.width, SD_SIZE):
          if tile_has_text_box(x, y, SD_SIZE, text_boxes):
            crop_x1 = min(x + SD_SIZE, image.height)
            crop_y1 = min(y + SD_SIZE, image.width)
            crop_box = (x, y, crop_x1, crop_y1)

            in_tile = pad_to_size(image.crop(crop_box), SD_SIZE)
            in_mask = pad_to_size(mask_image.crop(crop_box), SD_SIZE)

            out_tile = self.pipe(prompt=prompt, image=in_tile, mask_image=in_mask).images[0]
            out_tile = out_tile.crop((0, 0, crop_x1 - x, crop_y1 - y))
            out_image.paste(out_tile, (x, y))

    out_image.save(out_image_path)
