"""In-painting models."""
import io
import openai
import replicate
import requests
import tempfile

from PIL import Image, ImageDraw
from text_detector import TextBox
from typing import Sequence


class Inpainter:
  """Interface for in-painting models."""
  # TODO(julia): Run some experiments to determine the best prompt.
  PROMPT = "plain background"

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], out_image_path: str):
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
  SD_INPAINTING_V15 = "andreasjansson/stable-diffusion-inpainting"
  SD_INPAINTING_V15_VERSION = "8eb2da8345bee796efcd925573f077e36ed5fb4ea3ba240ef70c23cf33f0d848"

  def __init__(self, replicate_token: str, model_name=SD_INPAINTING_V15, model_version=SD_INPAINTING_V15_VERSION):
    replicate_client = replicate.Client(api_token=replicate_token)
    self.model = replicate_client.models.get(model_name).versions.get(model_version)

  @staticmethod
  def _make_mask(text_boxes: Sequence[TextBox], height: int, width: int, out_mask_path: str):
    """Returns a black image with white rectangles where the text boxes are."""
    mask = Image.new("RGB", (width, height), (0, 0, 0))  # black
    mask_draw = ImageDraw.Draw(mask)
    for text_box in text_boxes:
      mask_draw.rectangle(xy=(text_box.x, text_box.y, text_box.x + text_box.h, text_box.y + text_box.w),
                          fill=(255, 255, 255))  # white
    mask.save(out_mask_path)

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], out_image_path: str):
    image = Image.open(in_image_path)  # open the image to inspect its size
    mask_temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg")
    self._make_mask(text_boxes, image.height, image.width, mask_temp_file.name)

    url = self.model.predict(prompt=Inpainter.PROMPT,
                             image=open(in_image_path, "rb"),
                             mask=open(mask_temp_file, "rb"),
                             num_outputs=1)[0]

    out_image_data = requests.get(url).content
    out_image = Image.open(io.BytesIO(out_image_data))
    out_image.save(out_image_path)


class LocalInpainter(Inpainter):
  def __init__(self, inpainting_model_path: str):
    self.inpainting_model_path = inpainting_model_path

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], out_image_path: str):
    # TODO(julia): Implement this.
    raise NotImplemented()
