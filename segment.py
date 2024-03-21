import torch
import clip
import warnings
import numpy as np
import cv2 as cv

from PIL import Image

from utils import threshold_mask
warnings.filterwarnings("ignore")

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

class Segmenter:
    def __init__(self, device):
        self.device = device
        self._setModels()
    
    def _setModels(self):
        self.clipModel, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clipModel.eval()

        self.sam = sam_model_registry['vit_h'](checkpoint='checkpoints/sam_vit_h_4b8939.pth').to(self.device)
        self.predictor = SamAutomaticMaskGenerator(self.sam)
    
    def __call__(self, image, textPrompt):

        textfeatures = self._textFeatures(textPrompt)

        masks = self.predictor.generate(np.array(image))
        bboxs = [mask['bbox'] for mask in masks]
        masks = [mask['segmentation'] for mask in masks]
        masks = [np.array(mask).astype(np.uint8) for mask in masks]
        maskedImageFeatures = [mask[:,:,None] * np.array(image) for mask in masks]

        maskedImageFeatures = [self._imageFeatures(maskedImage) for maskedImage in maskedImageFeatures]

        similarityScores = [self._cosineSimilarity(textfeatures, maskedImageFeature) for maskedImageFeature in maskedImageFeatures]

        return masks[np.argmax(similarityScores)], bboxs[np.argmax(similarityScores)]

    def _textFeatures(self, text):
        return self.clipModel.encode_text(clip.tokenize([text]).to(self.device))
    
    def _imageFeatures(self, image):
        image_ = Image.fromarray(image.astype(np.uint8))
        image_ = self.preprocess(image_).unsqueeze(0).to(self.device)
        return self.clipModel.encode_image(image_)
    
    def _cosineSimilarity(self, f1, f2):
        return torch.nn.functional.cosine_similarity(f1, f2).item()
    