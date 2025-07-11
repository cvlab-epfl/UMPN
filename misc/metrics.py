import torch
import numpy as np
import motmetrics as mm
import pandas as pd

from ipdb import set_trace

from misc.log_utils import log

def make_dataframe_from_det(det_list, track_id_list):
    det_as_list = list()
    for frame_id, (frame_det, frame_track_ids) in enumerate(zip(det_list, track_id_list)):
        det_as_list.extend([{'FrameId':frame_id, 'Id':int(track_id), 'X':int(det[0]), 'Y':int(det[1])} 
                           for det, track_id in zip(frame_det, frame_track_ids)])

    det_as_df = pd.DataFrame(det_as_list)

    if  det_as_df.empty:
        det_as_df = pd.DataFrame(columns =['FrameId','Id','X','Y'])

    det_as_df = det_as_df.set_index(['FrameId', 'Id'])

    return det_as_df


def compute_mot_metric(gt_df, pred_df, metric_threshold):

    if gt_df.size == 0:
        print("Trying to compute tracking metric on an empty sequence (gt size is 0)")
        return None

    acc = mm.utils.compare_to_groundtruth(gt_df, pred_df, 'euc', distfields=['X', 'Y'], distth=metric_threshold)
    
    #library doesn't implement moda computation, compute it manually form accumulator
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')

    # print(summary)  # Uncomment for debugging
    metrics = dict(zip(summary.keys(), summary.values[0]))
    
    return metrics


def get_tracking_metrics(graph_data, pred_ids, gt_dict, metric_threshold):
    
    def merge_detections(world_points, track_ids, timestamps):
        unique_ids = torch.unique(track_ids)
        # log.debug(f"Pred world points: {world_points.shape} ")
        # log.debug(f"Pred track ids: {track_ids.shape} ")
        # log.debug(f"Pred timestamps: {timestamps.shape} ")
        # log.debug(f"Unique ids: {unique_ids} ")
        merged_world_points = []
        merged_track_ids = []
        merged_timestamps = []
        for uid in unique_ids:
            if uid == -1:  # Skip detections with no trajectory
                log.warning("Merging detections with no trajectory, this should not happen!")
                continue
            id_mask = track_ids == uid
            unique_timestamps = torch.unique(timestamps[id_mask])
            for ts in unique_timestamps:
                ts_mask = (track_ids == uid) & (timestamps == ts)
                avg_world_point = world_points[ts_mask].mean(dim=0)
                merged_world_points.append(avg_world_point)
                merged_track_ids.append(uid)
                merged_timestamps.append(ts)
        return torch.stack(merged_world_points) if merged_world_points else torch.empty(0, 3), torch.tensor(merged_track_ids), torch.tensor(merged_timestamps)
    
    world_points_pred = []
    world_points_gt = []
    track_ids_pred = []
    track_ids_gt = []
    timestamps_pred = []
    timestamps_gt = []
    
    for timestamp in sorted(gt_dict.keys()):
        # Ground truth data
        gt_world_points = []
        gt_track_ids = []
        gt_timestamps = []
        for view_id, view_data in gt_dict[timestamp].items():
            gt_world_points.append(torch.tensor(view_data['world_points']))
            gt_track_ids.append(torch.tensor(view_data['person_id']))
            gt_timestamps.append(torch.full((len(view_data['person_id']),), timestamp))
        
        gt_world_points = torch.cat(gt_world_points)
        gt_track_ids = torch.cat(gt_track_ids)
        gt_timestamps = torch.cat(gt_timestamps)
        
        # Merge ground truth detections from multiple views
        gt_world_points, gt_track_ids, gt_timestamps = merge_detections(gt_world_points, gt_track_ids, gt_timestamps)
        
        # Get predictions for this timestamp if they exist
        mask = graph_data['detection'].timestamp == timestamp
        if mask.any():
            pred_world_points = graph_data['detection'].world_points[mask]
            pred_track_ids = pred_ids[mask]
            pred_timestamps = graph_data['detection'].timestamp[mask]
            
            # Filter out detections with no trajectory (pred_ids == -1)
            valid_pred_mask = pred_track_ids != -1
            pred_world_points = pred_world_points[valid_pred_mask]
            pred_track_ids = pred_track_ids[valid_pred_mask]
            pred_timestamps = pred_timestamps[valid_pred_mask]
            
            if len(pred_world_points) != 0:
                pred_world_points, pred_track_ids, pred_timestamps = merge_detections(pred_world_points, pred_track_ids, pred_timestamps)
                
                world_points_pred.append(pred_world_points)
                track_ids_pred.append(pred_track_ids)
                timestamps_pred.append(pred_timestamps)
        
        world_points_gt.append(gt_world_points)
        track_ids_gt.append(gt_track_ids)
        timestamps_gt.append(gt_timestamps)


    # Concatenate all timesteps

    if len(world_points_pred) > 0:
        world_points_pred = torch.cat(world_points_pred, dim=0)
        track_ids_pred = torch.cat(track_ids_pred, dim=0)
        timestamps_pred = torch.cat(timestamps_pred, dim=0)
    else:
        world_points_pred = torch.empty((0, 3))
        track_ids_pred = torch.empty(0, dtype=torch.long)
        timestamps_pred = torch.empty(0, dtype=torch.long)

    try:
        world_points_gt = torch.cat(world_points_gt, dim=0)
        track_ids_gt = torch.cat(track_ids_gt, dim=0)
        timestamps_gt = torch.cat(timestamps_gt, dim=0)
    except Exception as e:
        import pdb; pdb.set_trace()
        
    # Compute metrics
    from misc.metrics import compute_mot_metrics_gp
    metrics = compute_mot_metrics_gp(
        world_points_pred.cpu().numpy(),
        world_points_gt.cpu().numpy(),
        track_ids_pred.cpu().numpy(),
        track_ids_gt.cpu().numpy(),
        timestamps_pred.cpu().numpy(),
        timestamps_gt.cpu().numpy(),
        metric_threshold
    )
    
    return metrics, (world_points_gt, track_ids_gt, timestamps_gt), (world_points_pred, track_ids_pred, timestamps_pred)


def compute_mot_metrics_gp(world_points_pred, world_points_gt, track_ids_pred, track_ids_gt, timestamps_pred, timestamps_gt, metric_threshold):
    """
    Compute MOT metrics for ground plane tracking using the new approach.

    Args:
    world_points_pred: (NUM_DETS, 3) Predicted world coordinates
    world_points_gt: (NUM_DETS, 3) Ground truth world coordinates
    track_ids_pred: (NUM_DETS) Predicted track IDs
    track_ids_gt: (NUM_DETS) Ground truth track IDs
    timestamps_pred: (NUM_DETS) Timestamps for each predicted detection
    timestamps_gt: (NUM_DETS) Timestamps for each ground truth detection

    Returns:
    dict: MOT metrics including MOTA, MOTP, IDF1, etc.
    """    
    # Get all unique timestamps
    all_timestamps = np.unique(np.concatenate([timestamps_pred, timestamps_gt]))
    
    # Organize ground truth detections and track IDs by frame
    gt_det_list = []
    gt_track_id_list = []
    for t in all_timestamps:
        mask_gt = timestamps_gt == t
        if np.any(mask_gt):
            # Use world coordinates directly without rescaling
            gt_world_points_t = world_points_gt[mask_gt]
            gt_track_ids_t = track_ids_gt[mask_gt]
            gt_det_list.append(gt_world_points_t[:, :2])
            gt_track_id_list.append(gt_track_ids_t)
        else:
            gt_det_list.append(np.empty((0, 2)))
            gt_track_id_list.append(np.empty(0, dtype=int))
    
    # Organize predicted detections and track IDs by frame
    pred_det_list = []
    pred_track_id_list = []
    for t in all_timestamps:
        mask_pred = timestamps_pred == t
        if np.any(mask_pred):
            # Use world coordinates directly without rescaling
            pred_world_points_t = world_points_pred[mask_pred]
            pred_track_ids_t = track_ids_pred[mask_pred]
            pred_det_list.append(pred_world_points_t[:, :2])
            pred_track_id_list.append(pred_track_ids_t)
        else:
            pred_det_list.append(np.empty((0, 2)))
            pred_track_id_list.append(np.empty(0, dtype=int))
    
    # Convert to DataFrames using helper function
    gt_df = make_dataframe_from_det(gt_det_list, gt_track_id_list)
    pred_df = make_dataframe_from_det(pred_det_list, pred_track_id_list)
    
    metrics = compute_mot_metric(gt_df, pred_df, metric_threshold)
    
    if metrics is None:
        # Return empty metrics dict if computation failed
        return {}
    
    return metrics
    
    
def calc_mot_metrics(accs):
    """
    This is a function for computing MOT metrics over many accumulators

    accs: A list of MOT accumulators

    Returns: (formatted string presenting MOT challenge metrics)
    [idf1 idp idr recall precision num_unique_objects mostly_tracked partially_tracked
    mostly_lost num_false_positives num_misses num_switches num_fragmentations mota motp]
    """
    # compute and display MOT metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        names=[str(x) for x in range(len(accs))],
        generate_overall=True,
    )

    return summary.to_dict("records")[-1]


def _compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def preprocess_bboxes_for_map(bbox_dict):
    """
    bbox_dict: {seq: (y_pred, bbox_pred)}
    y_pred: (frame, track_id)
    bbox_pred: (cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score)
    """
    res = dict()
    unique_ids = np.array([], dtype=str)
    unique_classes = np.array([], dtype=str)
    for seq, data in bbox_dict.items():
        y_pred, bbox_pred = data
        ids = np.array([seq + "_" + str(fr) for fr in y_pred[:, 0]], dtype=str)
        unique_ids = np.unique(np.concatenate((unique_ids, ids), 0))
        labels = bbox_pred[:, 0].astype(str)
        unique_classes = np.unique(np.concatenate((unique_classes, labels), 0))
        scores = bbox_pred[:, 13]
        xmin = bbox_pred[:, 2]
        ymin = bbox_pred[:, 3]
        xmax = bbox_pred[:, 4]
        ymax = bbox_pred[:, 5]

        for i, idx in enumerate(ids):
            idx = ids[i]
            label = labels[i]
            if idx not in res:
                res[idx] = dict()
            if label not in res[idx]:
                res[idx][label] = []
            box = [xmin[i], ymin[i], xmax[i], ymax[i], scores[i]]
            res[idx][label].append(box)
    return res, unique_ids, unique_classes



def calculate_classification_metrics(labels, preds, prefix=''):
    if labels.numel() == 0 or preds.numel() == 0:
        return {}

    if isinstance(preds, torch.Tensor) and preds.dim() > 1:
        preds = preds.squeeze()
    
    preds = (preds > 0.5).float()  # Assuming binary classification with threshold 0.5
    
    accuracy = (labels == preds).float().mean().item()
    
    # Confusion matrix
    true_positives = ((labels == 1) & (preds == 1)).sum().item()
    true_negatives = ((labels == 0) & (preds == 0)).sum().item()
    false_positives = ((labels == 0) & (preds == 1)).sum().item()
    false_negatives = ((labels == 1) & (preds == 0)).sum().item()
    
    # Precision, Recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    result = {
        f'{prefix}accuracy': accuracy,
        f'{prefix}precision': precision,
        f'{prefix}recall': recall,
        f'{prefix}f1_score': f1,
        f'{prefix}true_positives': true_positives,
        f'{prefix}true_negatives': true_negatives,
        f'{prefix}false_positives': false_positives,
        f'{prefix}false_negatives': false_negatives
    }

    return result