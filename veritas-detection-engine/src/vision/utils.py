import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image, size: int = 224) -> np.ndarray:
    """PIL image -> normalized NCHW float32 array (1, 3, size, size)."""
    img = image.convert("RGB").resize((size, size))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[None, ...]
