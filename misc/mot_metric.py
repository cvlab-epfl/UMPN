import trackeval
from pathlib import Path
import shutil
import numpy as np
import torch

from misc.log_utils import log

class MOTMetricEvaluator:
    def __init__(self, interpolate_missing_detections=False):
        self.initialized = False
        self.sequences = set()
        
        self.interpolate_missing_detections = interpolate_missing_detections
        
    def _initialize_directories(self, dset_name):
        """Initialize directory structure for trackeval"""
        self.root_track_eval_path = Path("TrackEvalDataTemp") / f"GNN_{dset_name}"
        self.root_track_eval_path.mkdir(exist_ok=True, parents=True)

        
        # Clear any existing files
        for item in self.root_track_eval_path.glob('*'):
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(str(item))
                
        # Create seqmap file with header
        seqmap_path = self.root_track_eval_path / f"gt/seqmaps/{dset_name}-seqmaps.txt"
        seqmap_path.parent.mkdir(parents=True, exist_ok=True)
        with open(seqmap_path, 'w') as f:
            f.write("name\n")
                
        self.initialized = True

    def save_sequence_data(self, graph_data, pred_ids, gt_dict, dset_name, sequence):
        """Save ground truth and predictions for a single sequence"""

        if not self.initialized:
            self._initialize_directories(dset_name)
            
        self.sequences.add((dset_name, sequence))
        
        if dset_name == "MOT17":
            detector = "-DPM"
        else:
            detector = ""

        log.info(f"Saving tracking data for {dset_name} {sequence}")

        # Prepare data structures for saving predictions and ground truth
        track_as_list_pred = []
        track_as_list_gt = []
        
        # Process each timestamp
        timestamps = sorted(gt_dict.keys())
        min_timestamp = min(timestamps)-1
        
        # Lists to store gt and pred info
        gt_world_points = []
        gt_track_ids = []
        gt_timestamps = []
        pred_world_points = []
        pred_track_ids = []
        pred_timestamps = []

        timestamp_to_skip = []
        frame_id = 1
        for timestamp in timestamps:
            # Convert timestamp to 0-based index
            # frame_id = int(timestamp - min_timestamp)
            
            skip_timestamp = False
            for view_id, view_data in gt_dict[timestamp].items():
                if len(view_data['bbox']) == 0:
                    # Add empty gt detection with confiddence 0 to validate the itmestamp
                    # track_as_list_gt.append((frame_id, -1, 0, 0, 0, 0, 0, 0, 0))
                    # skip_timestamp = True
                    continue
                
                for bbox, person_id, world_point in zip(view_data['bbox'], view_data['person_id'], view_data['world_points']):
                    track_as_list_gt.append((
                        frame_id,
                        int(person_id), 
                        float(bbox[0]), float(bbox[1]),
                        float(bbox[2]-bbox[0]), float(bbox[3]-bbox[1]),
                        1, 1, 1
                    ))
                    gt_world_points.append(torch.tensor(world_point))
                    gt_track_ids.append(torch.tensor(person_id))
                    gt_timestamps.append(torch.tensor(timestamp))
            
            if skip_timestamp:
                continue
            
            # Prediction data
            mask = graph_data['detection'].timestamp == timestamp
            if mask.any():
                batch_pred_bboxes = graph_data['detection'].bbox[mask]
                batch_pred_track_ids = pred_ids[mask]
                batch_pred_world_points = graph_data['detection'].world_points[mask]
                
                # Filter out detections with no trajectory
                valid_pred_mask = batch_pred_track_ids != -1
                batch_pred_bboxes = batch_pred_bboxes[valid_pred_mask]
                batch_pred_track_ids = batch_pred_track_ids[valid_pred_mask]
                batch_pred_world_points = batch_pred_world_points[valid_pred_mask]
                
                for bbox, track_id, world_point in zip(batch_pred_bboxes, batch_pred_track_ids, batch_pred_world_points):
                    track_as_list_pred.append({
                        'FrameId': frame_id,
                        'Id': int(track_id)+1,
                        'X': float(bbox[0]),
                        'Y': float(bbox[1]), 
                        'Width': float(bbox[2]-bbox[0]),
                        'Height': float(bbox[3]-bbox[1]),
                        'Score': 1.0
                    })
                    pred_world_points.append(world_point)
                    pred_track_ids.append(track_id)
                    pred_timestamps.append(torch.tensor(timestamp))
            
            frame_id += 1

        if self.interpolate_missing_detections:
            track_as_list_pred = interpolate_missing_detections(track_as_list_pred)

        # Save predictions
        pred_path = self.root_track_eval_path / f"pred/{dset_name}-seqmaps/GNNTracker/data/{dset_name}-{sequence}.txt"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, 'w') as f:
            for pred in track_as_list_pred:
                f.write(f"{pred['FrameId']},{pred['Id']},{pred['X']},{pred['Y']},{pred['Width']},{pred['Height']},{pred['Score']},-1,-1,-1\n")

        # Save ground truth
        gt_path = self.root_track_eval_path / f"gt/{dset_name}-seqmaps/{dset_name}-{sequence}/gt/gt.txt"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_path, 'w') as f:
            for gt in track_as_list_gt:
                f.write(f"{gt[0]},{gt[1]},{gt[2]},{gt[3]},{gt[4]},{gt[5]},{gt[6]},{gt[7]},{gt[8]}\n")

        # Read and save seqinfo.ini
        path_org_seqinfo = Path(f"/cvlabscratch/cvlab/home/engilber/datasets/{dset_name}/train/{dset_name}-{sequence}{detector}/seqinfo.ini")
        
        if path_org_seqinfo.exists():
            with open(path_org_seqinfo, 'r') as file:
                data = file.readlines()
        else:
            log.warning(f"No seqinfo.ini file found for {dset_name} {sequence}, using default values")
            data = [
                "[Sequence]\n",
                f"name={sequence}\n",
                "imDir=img1\n",
                "frameRate=30\n", 
                "seqLength=1\n",
                "imWidth=1920\n",
                "imHeight=1080\n",
                "imExt=.jpg\n"
            ]

        seqinfo_path = self.root_track_eval_path / f"gt/{dset_name}-seqmaps/{dset_name}-{sequence}/seqinfo.ini"
        seqinfo_path.parent.mkdir(parents=True, exist_ok=True)

        # Update seqLength in seqinfo
        new_seq_length = max(t[0] for t in track_as_list_gt)#len(set(t[0] for t in track_as_list_gt))
        new_data = []
        for line in data:
            if "seqLength" in line:
                new_data.append(f"seqLength={new_seq_length}\n")
            else:
                new_data.append(line)

        with open(seqinfo_path, 'w') as file:
            file.writelines(new_data)

        # Append sequence to seqmap
        seqmap_path = self.root_track_eval_path / f"gt/seqmaps/{dset_name}-seqmaps.txt"
        with open(seqmap_path, 'a') as f:
            f.write(f"{dset_name}-{sequence}\n")

        # Concatenate all timesteps
        gt_world_points = torch.stack(gt_world_points) if gt_world_points else torch.empty((0, 3))
        gt_track_ids = torch.stack(gt_track_ids) if gt_track_ids else torch.empty(0)
        gt_timestamps = torch.stack(gt_timestamps) if gt_timestamps else torch.empty(0)

        if len(pred_world_points) > 0:
            pred_world_points = torch.stack(pred_world_points)
            pred_track_ids = torch.stack(pred_track_ids)
            pred_timestamps = torch.stack(pred_timestamps)
        else:
            pred_world_points = torch.empty((0, 3))
            pred_track_ids = torch.empty(0)
            pred_timestamps = torch.empty(0)

        return (gt_world_points, gt_track_ids, gt_timestamps), (pred_world_points, pred_track_ids, pred_timestamps)

    def compute_metrics(self, dset_name):
        """Compute metrics for all saved sequences"""
        # Configure evaluation
        eval_config = trackeval.Evaluator.get_default_eval_config()
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5, 'PRINT_CONFIG': False}

        eval_config.update({
            'PRINT_ONLY_COMBINED': False,
            'PRINT_DETAILED': False,
            'PRINT_SUMMARY': False,
            'PRINT_CONFIG': False,
            'TIME_PROGRESS': False,
            'LOG_ON_ERROR': str(self.root_track_eval_path / "error.log"),
            'OUTPUT_DETAILED': False,
            'PRINT_RESULTS': False,
            'OUTPUT_SUMMARY': False,
            'PRINT_ONLY_COMBINED': True,
            'DISPLAY_LESS_PROGRESS': True,
            'PLOT_CURVES': False,
            'PRINT_CONFIG': False
        })

        dataset_config.update({
            'GT_FOLDER': str(self.root_track_eval_path / "gt"),
            'TRACKERS_FOLDER': str(self.root_track_eval_path / "pred"),
            'BENCHMARK': dset_name,
            'SPLIT_TO_EVAL': 'seqmaps',
            'TRACKERS_TO_EVAL': ['GNNTracker'],
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 1,
            'PRINT_CONFIG': False
        })

        # Run evaluation
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric(metrics_config))

        output_res, _ = evaluator.evaluate(dataset_list, metrics_list)

                # Flatten metrics for each sequence
        sequence_metrics = {}
        for seq_name, seq_data in output_res['MotChallenge2DBox']['GNNTracker'].items():
            flattened_metrics = {}
            for category in seq_data['pedestrian']:
                for metric, value in seq_data['pedestrian'][category].items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) or metric == 'HOTA':
                        if metric == 'HOTA':
                            value = np.mean(value)
                        if seq_name == 'COMBINED_SEQ':
                            flattened_metrics[f'combined_{metric}'] = value
                        else:
                            flattened_metrics[f'{seq_name}_{metric}'] = value
            sequence_metrics.update(flattened_metrics)

        # Clean up temporary files if needed
        # shutil.rmtree(self.root_track_eval_path)

        return sequence_metrics


def interpolate_missing_detections(track_list):
    """
    Interpolate missing detections for each unique track ID.
    For each track, if there are gaps in the frame IDs, linearly interpolate
    the bounding box coordinates for the missing frames.
    
    Args:
        track_list (List[dict]): List of track dictionaries containing FrameId, Id, X, Y, Width, Height
        
    Returns:
        List[dict]: Track list with interpolated detections added
    """
    # Group detections by track ID
    tracks_by_id = {}
    for det in track_list:
        track_id = det['Id']
        if track_id not in tracks_by_id:
            tracks_by_id[track_id] = []
        tracks_by_id[track_id].append(det)
    
    interpolated_tracks = []
    
    # Process each track
    for track_id, detections in tracks_by_id.items():
        # Sort by frame ID
        detections.sort(key=lambda x: x['FrameId'])
        
        # Get frame range
        start_frame = detections[0]['FrameId']
        end_frame = detections[-1]['FrameId']
        
        # Create dict of existing frames
        existing_frames = {d['FrameId']: d for d in detections}
        
        # Add original detections
        interpolated_tracks.extend(detections)
        
        # Interpolate missing frames
        for frame in range(start_frame + 1, end_frame):
            if frame not in existing_frames:
                # Find previous and next detections
                prev_frame = max(f for f in existing_frames.keys() if f < frame)
                next_frame = min(f for f in existing_frames.keys() if f > frame)
                
                prev_det = existing_frames[prev_frame]
                next_det = existing_frames[next_frame]
                
                # Calculate interpolation factor
                alpha = (frame - prev_frame) / (next_frame - prev_frame)
                
                # Interpolate bbox coordinates
                interpolated_det = {
                    'FrameId': frame,
                    'Id': track_id,
                    'X': prev_det['X'] + alpha * (next_det['X'] - prev_det['X']),
                    'Y': prev_det['Y'] + alpha * (next_det['Y'] - prev_det['Y']),
                    'Width': prev_det['Width'] + alpha * (next_det['Width'] - prev_det['Width']),
                    'Height': prev_det['Height'] + alpha * (next_det['Height'] - prev_det['Height']),
                    'Score': 1.0
                }
                
                interpolated_tracks.append(interpolated_det)
    
    # Sort by frame ID and track ID
    interpolated_tracks.sort(key=lambda x: (x['FrameId'], x['Id']))
    
    return interpolated_tracks
