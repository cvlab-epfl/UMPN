import json
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from collections import defaultdict

from misc.log_utils import log
from datasets.utils import merge_gt_with_det, process_crops

class SceneBaseSet():
    """
    This class is the base class for all scene based datasets.
    """

    def __init__(self, data_conf, training):
        super(SceneBaseSet, self).__init__()
        self.data_conf = data_conf
        
        self.training = training
        self.features = data_conf["features"]

        self.reid_type = data_conf["reid_type"]

        self.crops_size = data_conf["crops_size"]

        self.use_det = data_conf["use_detection_training"] if training else data_conf["use_detection_eval"]
        self.det_threshold = data_conf["detection_threshold"]
        self.iou_threshold = data_conf["iou_threshold_det_gt"]
        self.fill_missing_gt = data_conf["fill_missing_gt"] and training
        self.det_type = data_conf["detection_type"]
        
        if not hasattr(self, 'norm_dict_key'):
            self.norm_dict_key = f"{'with_det_' + self.det_type + '_' + str(self.det_threshold) if self.use_det else 'without_det'}"
            # self.norm_dict_key = f"{'with_det_' + self.det_type if self.use_det else 'without_det'}"

        log.info(f"Loading norm dict {self.gt_features_norm_path} using key {self.norm_dict_key}")
        self.norm_dict = self.load_norm_dict()
        self.frame_sizes = self._get_frame_sizes()


    def get(self, index, view_id):
        frame_dict = {}

        bbox, label, confidence, person_id, world_points, gt_dict = self.get_gt(index, view_id)
        frame_dict["bbox"] = torch.tensor(bbox, dtype=torch.float32)
        frame_dict["label"] = torch.tensor(label, dtype=torch.int)
        frame_dict["confidence"] = torch.tensor(confidence, dtype=torch.float32)
        frame_dict["person_id"] = torch.tensor(person_id, dtype=torch.int)
        frame_dict["world_points"] = torch.tensor(world_points, dtype=torch.float32)
        frame_dict["timestamp"] = torch.full((len(bbox),), index, dtype=torch.int)
        frame_dict["view_id"] = torch.full((len(bbox),), view_id, dtype=torch.int)
        frame_dict["gt_dict"] = gt_dict

        # crops
        if "crops" in self.features:
            crops = self.get_crops(index, view_id, bbox)

            # Normalize and resize crops
            crops = process_crops(crops, self.crops_size)
            # for crop in crops:
            #     log.info(f"Crop shape: {crop.shape}, Image ratio: {crop.shape[1] / crop.shape[0]:.2f}")
            frame_dict["crops"] = crops

        # ReID
        if "reid" in self.features:
            reid = self.get_reid(index, view_id, bbox)
            frame_dict["reid"] = torch.tensor(reid, dtype=torch.float32)

        # if "bbox_det" in self.features:
        #     bbox_det, bbox_scores, world_points_det = self.get_det(index, view_id)
        #     frame_dict["bbox_det"] = torch.tensor(bbox_det, dtype=torch.float32)
        #     frame_dict["bbox_scores"] = torch.tensor(bbox_scores, dtype=torch.float32)
        #     frame_dict["world_points_det"] = torch.tensor(world_points_det, dtype=torch.float32)

            # log.debug(f"Frame dict for index {index}, view {view_id}:")
            # log.debug(f"  bbox_det: shape {frame_dict['bbox_det'].shape}, dtype {frame_dict['bbox_det'].dtype}")
            # log.debug(f"  world_points_det: shape {frame_dict['world_points_det'].shape}, dtype {frame_dict['world_points_det'].dtype}")

            # exit()
        
        return frame_dict

    def get_norm_dict(self):
        return self.norm_dict.get(self.norm_dict_key, {})

    def load_norm_dict(self):
        norm_dict = defaultdict(dict)

        # Check if json file exists and load
        if self.gt_features_norm_path.exists():
            with open(self.gt_features_norm_path, 'r') as f:
                norm_dict.update(json.load(f))

        # Identify missing features that need normalization
        missing_features = [f for f in self.features if f not in norm_dict.get(self.norm_dict_key, {}) and self.features[f]['is_norm']]

        if missing_features or self.norm_dict_key not in norm_dict:
            norm_dict[self.norm_dict_key].update(self.compute_missing_feature_stats(missing_features))

            # Save updated norm_dict to json file
            with open(self.gt_features_norm_path, 'w') as f:
                json.dump(norm_dict, f)

        norm_dict[self.norm_dict_key] = {f: {
                        'mean': torch.tensor(v['mean']),
                        'std': torch.tensor(v['std'])
                    } for f, v in norm_dict[self.norm_dict_key].items() if f in self.features and self.features[f]['is_norm']}

        return norm_dict

    def compute_missing_feature_stats(self, missing_features):
        log.debug(f"Some statistic are missing for features {missing_features}, computing them: this may take a while...")
        feature_values = {f: [] for f in missing_features}
        
        # Compute stats over all dataset in a single pass
        total_iterations = self.__len__() * len(self.get_views())
        with tqdm(total=total_iterations, desc="Computing feature stats") as pbar:
            for index in range(self.__len__()):
                for view_id in self.get_views():
                    frame_dict = self.get(index, view_id)

                    #get crops and reid if needed
                    if "crops" in missing_features:
                        crops = self.get_crops(index, view_id, frame_dict["bbox"])
                        frame_dict["crops"] = crops

                    if "reid" in missing_features:
                        reid = self.get_reid(index, view_id, frame_dict["bbox"])
                        frame_dict["reid"] = reid

                    for feature in missing_features:
                        if feature in frame_dict:
                            feature_values[feature].extend(frame_dict[feature])
                    
                    pbar.update(1)

        # Compute mean and std for each missing feature
        norm_dict = {}

        for feature in missing_features:
            values = np.vstack(feature_values[feature])
            log.debug(f"feature: {feature}, values: {values.shape}")
            norm_dict[feature] = {
                "mean": np.mean(values, axis=0).tolist(),
                "std": np.std(values, axis=0).tolist()
            }

        log.debug(f"norm_dict: {norm_dict}")

        return norm_dict

    def get_gt(self, index, view_id):
        bbox, person_id, world_points = self._get_gt(index, view_id)

        gt_dict = {
            "bbox": bbox,
            "person_id": person_id,
            "world_points": world_points,
            "timestamp": index,
            "view_id": view_id
        }
        
        label = np.ones(len(bbox), dtype=int)
        confidence = np.ones(len(bbox), dtype=np.float32)

        if self.use_det:
            bbox_det, bbox_scores, world_points_det = self.get_det(index, view_id)
            bbox, label, confidence, person_id, world_points = merge_gt_with_det(bbox, world_points, label, person_id, bbox_det, bbox_scores, world_points_det, iou_threshold=self.iou_threshold, include_unmatched_gt=self.training and self.fill_missing_gt)
            
        return bbox, label, confidence, person_id, world_points, gt_dict

    def _get_gt(self, index, view_id):
        log.error("Abstract class get_gt as been called, it should be overwriten it in child class")
        raise NotImplementedError

    def get_crops(self, index, view_id, bbox):
        log.error("Abstract class get_crops as been called, it should be overwriten it in child class")
        raise NotImplementedError

    def get_reid(self, index, view_id, bbox):
        # Determine the appropriate path based on whether detections are used
        if self.use_det:
            if self.fill_missing_gt:
                reid_path = self.gt_reid_path / self.reid_type / f"{self.det_type}_{self.det_threshold}_fill_missing_gt" / f"{index}_{view_id}.npy"
            else:
                reid_path = self.gt_reid_path / self.reid_type / f"{self.det_type}_{self.det_threshold}" / f"{index}_{view_id}.npy"
        else:
            reid_path = self.gt_reid_path / self.reid_type / f"{index}_{view_id}.npy"

        if reid_path.exists():
            reid = np.load(reid_path)
        else:
            log.warning(f"Precomputed reid features not found at {reid_path}")
            from datasets.reid import generate_and_save_reid
            reid = generate_and_save_reid(self.reid_type, index, view_id, self.get_crops(index, view_id, bbox), reid_path)

        return reid

    def get_det(self, index, view_id):

        det_path = self.gt_det_path / self.det_type / f"{index}_{view_id}.npy"

        if det_path.exists():
            saved_array = np.load(det_path)
            bbox_det = saved_array[:, :4]
            bbox_scores = saved_array[:, 4]
        else:
            log.warning(f"Precomputed det features not found at {det_path}")
            from datasets.detection import generate_and_save_det
            bbox_det, bbox_scores = generate_and_save_det(self.det_type, index, view_id, self.get_frame(index, view_id), det_path)

        # Check if bbox_scores is nx1, if not, reshape it
        if bbox_scores.ndim == 1:
            bbox_scores = bbox_scores.reshape(-1, 1)

        # Filter detections based on detection threshold
        mask = bbox_scores[:, 0] >= self.det_threshold
        bbox_det = bbox_det[mask]
        bbox_scores = bbox_scores[mask]

        # Add debug log if some bboxes are being masked
        num_masked = np.sum(~mask)
        if num_masked > 0:
            log.spam(f"Masked {num_masked} detections with scores below threshold {self.det_threshold} for frame {index}, view {view_id}")

        return bbox_det, bbox_scores

    def get_camera_dict(self, index):
        camera_dict = self._get_camera_dict(index)
        for view_id, camera_info in camera_dict.items():
            # Convert all data to float tensors
            camera_info['position'] = torch.tensor(camera_info['position'], dtype=torch.float)
            camera_info['rotation'] = torch.tensor(camera_info['rotation'], dtype=torch.float)
            camera_info['translation'] = torch.tensor(camera_info['translation'], dtype=torch.float)
            camera_info['intrinsic'] = torch.tensor(camera_info['intrinsic'], dtype=torch.float)
            camera_info['frame_size'] = torch.tensor(camera_info['frame_size'], dtype=torch.float)
            camera_info['axis'] = torch.tensor(camera_info['axis'], dtype=torch.float)
            
            # Compute a hash for each camera
            camera_hash = hash(tuple(map(tuple, [
                camera_info['position'].tolist(),
                camera_info['rotation'].flatten().tolist(),
                camera_info['translation'].flatten().tolist(),
                camera_info['intrinsic'].flatten().tolist(),
                camera_info['frame_size'].tolist(),
                camera_info['axis'].tolist()
            ])))
            camera_info['hash'] = torch.tensor([camera_hash], dtype=torch.long)
            
        return camera_dict

    def _get_frame_sizes(self):
        frame_sizes = {}

        for view_id in self.get_views():
            frame = self.get_frame(0, view_id)
            if frame is not None:
                frame_sizes[view_id] = frame.shape[:2]
            else:
                log.warning(f"Unable to get frame for view {view_id}")

        if not frame_sizes:
            log.error("No valid frames found for any view")
            return None

        return frame_sizes

    def get_frame(self, index, view_id):
        log.error("Abstract class get_frame as been called, it should be overwriten it in child class")
        raise NotImplementedError

    def get_length(self):
        log.error("Abstract class get_lentgth as been called, it should be overwriten it in child class")
        raise NotImplementedError

    def get_views(self):
        log.error("Abstract class get_views as been called, it should be overwriten it in child class")
        raise NotImplementedError

    def __len__(self):
        length = int(self.get_length())
        return length