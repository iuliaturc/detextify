"""Interfaces for text detection."""
from dataclasses import dataclass
from typing import Sequence


@dataclass
class TextBox:
  x: int
  y: int
  w: int
  h: int


class TextDetector:
  def detect_text(self, image_filename: str) -> Sequence[TextBox]:
    pass
