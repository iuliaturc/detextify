"""In-painting models."""
import io
import openai
import requests

from PIL import Image, ImageDraw
from text_detector import TextBox
from typing import Sequence


class Inpainter:
  """Interface for in-painting models."""
  # TODO(julia): Run some experiments to determine the best prompt.
  PROMPT = "plain background"

  def inpaint(self, image_path: str, text_boxes: Sequence[TextBox], out_image_path: str) -> str:
    pass


class DalleInpainter(Inpainter):
  """In-painting model that calls the DALL-E API."""

  def __init__(self, openai_key: str):
    openai.api_key = openai_key

  @staticmethod
  def _make_mask(text_boxes: Sequence[TextBox], height: int, width: int) -> bytes:
    mask = Image.new("RGBA", (width, height), (0, 0, 0, 1))  # fully opaque
    mask_draw = ImageDraw.Draw(mask)
    for text_box in text_boxes:
      mask_draw.rectangle(xy=(text_box.x, text_box.y, text_box.x + text_box.h, text_box.y + text_box.w),
                          fill=(0, 0, 0, 0))  # fully transparent
    # Convert mask to bytes.
    bytes_arr = io.BytesIO()
    mask.save(bytes_arr, format="PNG")
    return bytes_arr.getvalue()

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], out_image_path: str):
    image = Image.open(in_image_path)  # open the image to inspect its size

    response = openai.Image.create_edit(
        image=open(in_image_path, "rb"),
        mask=self._make_mask(text_boxes, image.height, image.width),
        prompt=Inpainter.PROMPT,
        n=1,
        size=f"{image.height}x{image.width}"
    )
    url = response['data'][0]['url']
    out_image_data = requests.get(url).content
    out_image = Image.open(io.BytesIO(out_image_data))
    out_image.save(out_image_path)


class ReplicateInpainter(Inpainter):
  def __init__(self, replicate_key: str):
    self.replicate_key = replicate_key

  def inpaint(self, image_path: str, text_boxes: Sequence[TextBox], out_image_path: str) -> str:
    # TODO(julia): Implement this.
    raise NotImplemented()


class LocalInpainter(Inpainter):
  def __init__(self, inpainting_model_path: str):
    self.inpainting_model_path = inpainting_model_path

  def inpaint(self, image_path: str, text_boxes: Sequence[TextBox], out_image_path: str) -> str:
    # TODO(julia): Implement this.
    raise NotImplemented()
