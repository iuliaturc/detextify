"""In-painting models."""
import io
import math
import openai
import replicate
import requests
import tempfile
import torch

from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline
from typing import Sequence

import detextify.utils as utils
from detextify.text_detector import TextBox


class Inpainter:
  """Interface for in-painting models."""
  # TODO(julia): Run some experiments to determine the best prompt.
  DEFAULT_PROMPT = "plain background"

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], prompt: str, out_image_path: str):
    pass


class DalleInpainter(Inpainter):
  """In-painting model that calls the DALL-E API."""

  def __init__(self, openai_key: str):
    openai.api_key = openai_key

  @staticmethod
  def _make_mask(text_boxes: Sequence[TextBox], height: int, width: int) -> bytes:
    """Returns an .png where the text boxes are transparent."""
    mask = Image.new("RGBA", (width, height), (0, 0, 0, 1))  # fully opaque
    mask_draw = ImageDraw.Draw(mask)
    for text_box in text_boxes:
      mask_draw.rectangle(xy=(text_box.x, text_box.y, text_box.x + text_box.h, text_box.y + text_box.w),
                          fill=(0, 0, 0, 0))  # fully transparent
    # Convert mask to bytes.
    bytes_arr = io.BytesIO()
    mask.save(bytes_arr, format="PNG")
    return bytes_arr.getvalue()

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], prompt: str, out_image_path: str):
    image = Image.open(in_image_path)  # open the image to inspect its size

    response = openai.Image.create_edit(
        image=open(in_image_path, "rb"),
        mask=self._make_mask(text_boxes, image.height, image.width),
        prompt=prompt,
        n=1,
        size=f"{image.height}x{image.width}"
    )
    url = response['data'][0]['url']
    out_image_data = requests.get(url).content
    out_image = Image.open(io.BytesIO(out_image_data))
    out_image.save(out_image_path)


class StableDiffusionInpainter(Inpainter):
  """Abstract class for Stable Diffusion inpainters; suppoerts any input image size. Children must implement `call_model`."""

  def call_model(self, prompt: str, image: Image, mask: Image) -> Image:
    pass  # To be implemented by children.

  def _tile_has_text_box(self, crop_x: int, crop_y: int, crop_size: int, text_boxes: Sequence[TextBox]):
    # Turn the tile into a TextBox just so that we can reuse utils.boxes_intersect
    crop_box = TextBox(crop_x, crop_y, crop_size, crop_size)
    return any([utils.boxes_intersect(crop_box, text_box) for text_box in text_boxes])

  def _pad_to_size(self, image, size):
    new_image = Image.new(image.mode, (size, size), color=(0, 0, 0))
    new_image.paste(image)
    return new_image

  def _make_mask(self, text_boxes: Sequence[TextBox], height: int, width: int, mode: str) -> Image:
    """Returns a black image with white rectangles where the text boxes are."""
    num_channels = len(mode)
    background_color = tuple([0] * num_channels)
    mask_color = tuple([255] * num_channels)

    mask = Image.new(mode, (width, height), background_color)
    mask_draw = ImageDraw.Draw(mask)
    for text_box in text_boxes:
      mask_draw.rectangle(xy=(text_box.x, text_box.y, text_box.x + text_box.h, text_box.y + text_box.w),
                          fill=mask_color)
    return mask

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], prompt: str, out_image_path: str):
    image = Image.open(in_image_path)
    mask_image = self._make_mask(text_boxes, image.height, image.width, image.mode)

    # SD only accepts images that are exactly 512 x 512.
    SD_SIZE = 512

    if image.height == SD_SIZE and image.width == SD_SIZE:
      out_image = self.call_model(prompt=prompt, image=image, mask=mask_image)
    else:
      # Break the image into 512 x 512 tiles. In-paint the tiles that contain text boxes.
      out_image = image.copy()

      # Used for the final out_image.paste; required to be in mode L.
      mask_binary = self._make_mask(text_boxes, image.height, image.width, "L")

      for x in range(0, image.height, SD_SIZE):
        for y in range(0, image.width, SD_SIZE):
          if self._tile_has_text_box(x, y, SD_SIZE, text_boxes):
            crop_x1 = min(x + SD_SIZE, image.height)
            crop_y1 = min(y + SD_SIZE, image.width)
            crop_box = (x, y, crop_x1, crop_y1)

            in_tile = self._pad_to_size(image.crop(crop_box), SD_SIZE)
            in_mask = self._pad_to_size(mask_image.crop(crop_box), SD_SIZE)
            out_tile = self.call_model(prompt=prompt, image=in_tile, mask=in_mask)
            out_tile = out_tile.crop((0, 0, crop_x1 - x, crop_y1 - y))
            out_mask = mask_binary.crop(crop_box)
            out_image.paste(out_tile, (x, y), out_mask)

    out_image.save(out_image_path)


class ReplicateSDInpainter(StableDiffusionInpainter):
  SD_INPAINTING_V2 = "cjwbw/stable-diffusion-v2-inpainting"
  SD_INPAINTING_V2_VERSION = "f9bb0632bfdceb83196e85521b9b55895f8ff3d1d3b487fd1973210c0eb30bec"

  def __init__(self, replicate_token: str, model_name=SD_INPAINTING_V2, model_version=SD_INPAINTING_V2_VERSION):
    replicate_client = replicate.Client(api_token=replicate_token)
    self.model = replicate_client.models.get(model_name).versions.get(model_version)

  def call_model(self, prompt: str, image: Image, mask: Image) -> Image:
    # Replicate expects a file object as an input.
    img_temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg")
    image.save(img_temp_file)
    mask_temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg")
    mask.save(mask_temp_file)

    url = self.model.predict(prompt=prompt,
                             prompt_strength=1.0,
                             image=open(img_temp_file.name, "rb"),
                             mask=open(mask_temp_file.name, "rb"),
                             num_outputs=1)[0]
    out_image_data = requests.get(url).content
    out_image = Image.open(io.BytesIO(out_image_data))
    return out_image


class LocalSDInpainter(StableDiffusionInpainter):
  """Uses a local Stable Diffusion model from HuggingFace for in-painting."""

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

  def call_model(self, prompt: str, image: Image, mask: Image) -> Image:
    return self.pipe(prompt=prompt, image=image, mask_image=mask).images[0]

