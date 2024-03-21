import torch

import warnings
import numpy as np
import cv2 as cv

from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

warnings.filterwarnings("ignore")

class inPainter:
    def __init__(self, device):
        self.device = device
        self._setModels()

    def _setModels(self):
        self.inPaintPipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        ).to(self.device)

    def _dilation(self, mask, kernel_size=7):
        kernel = np.ones((kernel_size, kernel_size))
        mask = np.array(mask)
        mask = cv.dilate(mask, kernel, iterations=2)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)
        return mask

    def __call__(self, image, mask, prompt):
        mask = self._dilation(mask)
        prompt_ = "Empty Void"
        im = self.inPaintPipe(prompt=prompt_, image=image, mask_image=mask, negative_prompt=prompt+" objects").images[0]
        im.resize((image.size[0], image.size[1]))
        return im
