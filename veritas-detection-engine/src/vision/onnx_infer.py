import numpy as np
import onnxruntime as ort
from PIL import Image

from src.vision.utils import preprocess


class ImageArtifactDetector:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_ai_probability(self, image: Image.Image) -> float:
        logits = self.session.run(None, {self.input_name: preprocess(image)})[0][0]
        exp = np.exp(logits - logits.max())
        return float((exp / exp.sum())[1])  # class 1 = AI-generated
