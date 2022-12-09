"""Interfaces for text detection."""
from dataclasses import dataclass
from typing import Sequence


@dataclass
class TextBox:
  # (x, y) is the top left corner of a rectangle; the origin of the coordinate system is the top-left of the image.
  # x denotes the vertical axis, y denotes the horizontal axis (to match the traditional indexing in a matrix).
  x: int
  y: int
  h: int
  w: int
  text: str = None


class TextDetector:
  def detect_text(self, image_filename: str) -> Sequence[TextBox]:
    pass
