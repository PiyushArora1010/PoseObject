import sys
sys.path.append('nvs')

from nvs.utils.zero123_utils import init_model
from nvs.utils.sam_utils import sam_init
from nvs.view import view

def generate_view(input_raw, xs, ys, device):
    models = init_model(device, 'nvs/zero123-xl.ckpt', half_precision=True)
    model_zero123 = models["turncam"]
    predictor = None
    return view(input_raw, xs, ys, predictor, model_zero123, device)
