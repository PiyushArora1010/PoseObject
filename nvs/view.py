import os
import numpy as np
import torch
from PIL import Image
from nvs.utils.zero123_utils import  sample_model_batch
from nvs.utils.sam_utils import sam_out_nosave
from nvs.utils.utils import pred_bbox, image_preprocess_nosave
from nvs.ldm.models.diffusion.ddim import DDIMSampler
from torchvision import transforms


def preprocess(predictor, raw_im, lower_contrast=False):
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), pred_bbox(raw_im))
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256


def view(input_raw, xs, ys, predictor, model_zero123, device):
    #input_256 = preprocess(predictor, input_raw)
    h = 256
    w = 256
    input_256 = input_raw.resize((h,w))
    input_im_init = np.asarray(input_256, dtype=np.float32) / 255.0
    input_im = transforms.ToTensor()(input_im_init).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    
    n_samples = 1
    xs = [xs]
    ys = [ys]

    sampler = DDIMSampler(model_zero123)

    x_samples_ddims = sample_model_batch(model_zero123, sampler, input_im, xs, ys, n_samples=n_samples, h=h, w=w)
    x_samples_ddims = x_samples_ddims[0].cpu().numpy()
    x_samples_ddims = np.transpose(x_samples_ddims, (1, 2, 0))
    x_samples_ddims = 255.0 * x_samples_ddims
    out_image = Image.fromarray(x_samples_ddims.astype(np.uint8))
    return out_image
