import numpy as np
import torch
import torchreid

from configs.pathes import model_path
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
from misc.log_utils import log

USE_CUDA_FOR_REID = True
log.warning(f"Loading reid model {'on GPU' if USE_CUDA_FOR_REID else 'on CPU'}")

osnet_model = torchreid.models.osnet_ain.osnet_ain_x1_0(2510)
checkpoint = torch.load(model_path["osnet_ain_ms_d_c"], map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']
# Remove the 'module.' prefix from keys
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
osnet_model.load_state_dict(state_dict)
osnet_model.eval()

if USE_CUDA_FOR_REID:
    osnet_model = osnet_model.cuda()

reid_model = {
    "osnet": osnet_model
}

height = 256
width = 128

norm_mean = [0.485, 0.456, 0.406] # imagenet mean
norm_std = [0.229, 0.224, 0.225] # imagenet std
normalize = Normalize(mean=norm_mean, std=norm_std)
    
transform_te = Compose([
    ToPILImage(),
    Resize((height, width)),
    ToTensor(),
    normalize,
])

def generate_and_save_reid(reid_type, index, view_id, crops, reid_path):

    if reid_type == "osnet":
        model = reid_model["osnet"]
    else:
        raise NotImplementedError(f"ReID type {reid_type} not implemented")
        
    # Crops might have different sizes, so we need to resize them to the same size
    
    resized_crops = []
    for crop in crops:
        try:
            resized_crop = transform_te(crop)
        except Exception as e:
            log.warning(f"Failed to transform crop: {e}")
            # Create a dummy crop of zeros with correct shape
            resized_crop = torch.zeros((3, height, width))
        resized_crops.append(resized_crop)
        
    tensor_crops = torch.stack(resized_crops)

    if USE_CUDA_FOR_REID:
        tensor_crops = tensor_crops.cuda()

    with torch.no_grad():
        features = model(tensor_crops)

    features = features.cpu().numpy()

    # Ensure the parent directory exists
    reid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(reid_path, features)

    return features
        



        
