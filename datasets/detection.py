import numpy as np
from mmdet.apis import DetInferencer

from configs.pathes import model_path
from misc.log_utils import log

USE_CUDA_FOR_DET = True
log.warning(f"Loading detection models {'on GPU' if USE_CUDA_FOR_DET else 'on CPU'}")

device = 'cuda:0' if USE_CUDA_FOR_DET else 'cpu'

# Load the RTMDet model
config_file_rtmdet = model_path["config_rtmdet"]
checkpoint_file_rtmdet = model_path["checkpoint_rtmdet"]
inferencer_rtmdet = DetInferencer(model=str(config_file_rtmdet), weights=str(checkpoint_file_rtmdet), device=device)

# Load the YOLOX model
config_file_yolox = model_path["config_yolox"]
checkpoint_file_yolox = model_path["checkpoint_yolox"]
inferencer_yolox = DetInferencer(model=str(config_file_yolox), weights=str(checkpoint_file_yolox), device=device)

def generate_and_save_det(det_type, index, view_id, frame, det_path):
    if det_type not in ["rtmdet", "yolox", "yolox_bytetrack"]:
        raise NotImplementedError(f"Detection type {det_type} not implemented")

    # Run inference
    if det_type == "rtmdet":
        result = inferencer_rtmdet(frame, return_datasamples=True)
    elif det_type == "yolox":
        result = inferencer_yolox(frame, return_datasamples=True)

    # Process the results (assuming we're only interested in person detections)
    person_class_id = 0  # Adjust this if needed based on your model's class mapping
    pred_instances = result['predictions'][0].pred_instances.numpy()
    bboxes = pred_instances.bboxes
    scores = pred_instances.scores
    labels = pred_instances.labels

    # Filter detections based on confidence threshold and class
    confidence_threshold = 0.1
    mask = (scores > confidence_threshold) & (labels == person_class_id)
    high_conf_bboxes = bboxes[mask]
    high_conf_scores = scores[mask]

    # Combine bboxes and scores
    detections = np.hstack((high_conf_bboxes, high_conf_scores.reshape(-1, 1)))

    # Ensure the parent directory exists
    det_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(det_path, detections)

    return high_conf_bboxes, high_conf_scores