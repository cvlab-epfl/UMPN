import json

import numpy as np

from collections import defaultdict

from configs.pathes import data_path
from datasets.utils import Calibration, Annotations, get_frame_from_file
from datasets.sceneset import SceneBaseSet
from misc import geometry
from misc.log_utils import log



class ScoutSet(SceneBaseSet):
    def __init__(self, data_conf, training, sequence_name, single_cam=None, cam_name=None):

        self.root_path = data_path['scout_root'] 
        
        #/ sequence_name

        self.dset_name = "SCOUT"

        if single_cam is None:
            self.sequence = sequence_name
        else:
            self.sequence = f"{sequence_name}_cam{single_cam}"

        self.frame_dir_path = self.root_path / "images" / sequence_name
        self.gt_dir_path = self.root_path / "annotations/individual_format/" / sequence_name
        self.calib_dir_path = self.root_path / "calibrations"
        
        self.gt_reid_path = self.root_path / "annotations_reid/" / sequence_name
        self.gt_det_path = self.root_path / "pretrained_detections/" / sequence_name

        if single_cam is not None:
            self.cam_name = [single_cam]
        elif cam_name is not None:
            self.cam_name = cam_name
        else:
            self.cam_name = [int(folder.name.replace("cam_","")) for folder in self.frame_dir_path.iterdir() if folder.is_dir() and folder.name.startswith("cam_")]


        self.gt_features_norm_path = self.root_path / "annotations_features_norm.json"

        self.nb_frames = len([frame_path for frame_path in (self.frame_dir_path / "cam_0").iterdir() if frame_path.suffix == ".jpg"])
        self.calibs = load_calibrations(self.calib_dir_path, single_cam)

        mesh_path = data_path['scout_root'] / "meshes" / "mesh_ground_only.ply"
        self.scene_mesh = geometry.load_mesh(mesh_path)

        # Print max index for each camera in annotations_dict
        # for cam_id in self.annotations_dict.keys():
        #     max_frame_id = max(self.annotations_dict[cam_id].keys()) if self.annotations_dict[cam_id] else -1
        #     log.debug(f"Camera {cam_id}: Max frame index = {max_frame_id}, number of annotations = {len(self.annotations_dict[cam_id])}")

        # log.debug(f"Annotations dict keys: {self.annotations_dict.keys()}")
        # log.debug(f"Annotations dict total number of annotations: {sum(len(annotations) for annotations in self.annotations_dict.values())}")

        use_det = data_conf["use_detection_training"] if training else data_conf["use_detection_eval"] 
        self.norm_dict_key = f"{'with_det_' + data_conf['detection_type'] if use_det else 'without_det'}"

        super().__init__(data_conf, training)


        log.debug(f"Dataset SCOUT containing {self.nb_frames} with views {self.get_views()}")


    def get_frame(self, index, view_id):
            """
            Read and return undistoreted frame coresponding to index and view_id.
            The frame is return at the original resolution
            """

            frame_path = self.frame_dir_path / f"cam_{view_id}/image_{index}.jpg"

            # log.debug(f"pomelo dataset get frame {index} {view_id}")
            frame = get_frame_from_file(frame_path)

            return frame

    def _get_gt(self, index, view_id):
        gt_file_path = self.gt_dir_path / f"cam_{view_id}" / f"image_{index}.txt"

        annotations = _load_gt_from_file(gt_file_path, index, view_id)

        bboxes = []
        person_ids = []
        world_points = []

        if len(annotations) == 0:
            # log.warning(f"No annotations found for frame {index}, view {view_id}")
            return np.array([]), np.array([], dtype=np.int32), np.array([])

        for annotation in annotations:
            # Convert bbox from x,y,w,h to x1,y1,x2,y2 format
            x, y, w, h = annotation["bbox"]["x"], annotation["bbox"]["y"], annotation["bbox"]["w"], annotation["bbox"]["h"]
            bbox = [x, y, x + w, y + h]
            
            bboxes.append(bbox)
            person_ids.append(annotation["track_id"])
            world_points.append([annotation["world"]["Xw"], annotation["world"]["Yw"], annotation["world"]["Zw"]])

        return np.stack(bboxes), np.stack(person_ids).astype(np.int32), np.stack(world_points)

    def get_det(self, index, view_id, frame_height=1080):

        #call parent function to get det bboxes
        det_bboxes, det_scores = super().get_det(index, view_id)

        # Load the average aspect ratio from self.norm_dict
        # Aspect ratio is width / height
        if hasattr(self, 'norm_dict') and 'without_det' in self.norm_dict and 'bbox' in self.norm_dict['without_det']:
            bbox_mean = self.norm_dict['without_det']['bbox']['mean']
            avg_width = bbox_mean[2] - bbox_mean[0]
            avg_height = bbox_mean[3] - bbox_mean[1]
            avg_aspect_ratio = avg_width / avg_height
        else:
            # Default aspect ratio if norm_dict not available
            avg_aspect_ratio = 0.35  # Typical person bbox aspect ratio (width/height)
        
        det_world_points = []
        
        for bbox in det_bboxes:
            x1, y1, x2, y2 = bbox
            xc = (x1 + x2) / 2  # x-center of bbox
            
            # Check if the bottom of the bbox is close to the bottom of the image
            if frame_height - y2 < 10:  # Assuming 10 pixels as threshold
                # Correct the bottom point using the average aspect ratio
                bbox_width = x2 - x1
                corrected_height = bbox_width / avg_aspect_ratio
                corrected_y2 = y1 + corrected_height
                
                # log.debug(f"Corrected bbox bottom from {y2} to {corrected_y2} at index {index}, view {view_id}")
                
                bottom_center = np.array([[xc, corrected_y2]])
            else:
                bottom_center = np.array([[xc, y2]])
            
            # Use the calibration for the current view
            calib = self.calibs[view_id]
            
            # Project bottom center to world coordinates using mesh intersection
            world_point = np.vstack([geometry.project_2d_points_to_mesh([fp], calib, self.scene_mesh) for fp in bottom_center]).reshape(-1, 3)
            
            det_world_points.append(world_point.squeeze())
        
        det_world_points = np.array(det_world_points)

        return det_bboxes, det_scores, det_world_points
    
    def get_crops(self, index, view_id, bboxes):
        index = index * 10

        frame = self.get_frame(index, view_id)
        height, width = frame.shape[:2]

        crops = []
        for i, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = bbox
            x1, y1 = max(0, xmin), max(0, ymin)
            x2, y2 = min(width, xmax), min(height, ymax)
            
            if x1 >= x2 or y1 >= y2:
                # Bbox is completely outside the image
                log.warning(f"Bbox {i} at index {index} is outside the image. Returning None.")
                crops.append(None)
            else:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)

        return crops

    def _get_camera_dict(self, index):
        camera_dict = {}
        for view_id in self.get_views():
            calib = self.calibs[view_id]
            
            # Compute camera position
            R_inv = np.linalg.inv(calib.R)
            camera_position = -R_inv @ calib.T
            
            # Compute camera axis (normal to the camera plane)
            camera_axis = R_inv[:, 2]  # The third column of R_inv is the camera's z-axis (normal to image plane)
            
            camera_dict[view_id] = {
                'position': camera_position.flatten(),
                'rotation': calib.R,
                'translation': calib.T,
                'intrinsic': calib.K,
                'frame_size': self.frame_sizes[view_id],
                'axis': camera_axis
            }
        return camera_dict

    def get_views(self):
        return self.cam_name
        
    def get_length(self):
        return self.nb_frames



def load_calib(calib_file_path):
    
    with open(calib_file_path, 'r') as f:
        calib_dict = json.load(f)

    if "K" in calib_dict:
        K =  np.array(calib_dict["K"])
    else:
        K = None

    if "R" in calib_dict:
        R =  np.array(calib_dict["R"])
    else:
        R = None

    if "T" in calib_dict:
        T =  np.array(calib_dict["T"])
    else:
        T = None

    if "dist" in calib_dict:
        dist =  np.array(calib_dict["dist"])
    else:
        dist = None

    if "view_id" in calib_dict:
        view_id =  calib_dict["view_id"]
    else:
        view_id = str(calib_file_path)
        
    calib = Calibration(K, R, T, dist, view_id)

    return calib

        
# Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'dist', 'view_id'])
def load_calibrations(calib_dir_path, single_cam=None):
    
    calib_file_pattern = "cam_*.json" if single_cam is None else f"cam_{single_cam}.json"
    calibs = {}
    for calib_file in calib_dir_path.glob(calib_file_pattern):
        # Extract camera number from filename (e.g. "cam_10.json" -> 10)
        cam_num = int(calib_file.stem.replace("cam_", ""))
        
        # Load calibration for this camera
        calib = load_calib(calib_file)
        
        # Store in dictionary with camera number as key
        calibs[cam_num] = calib

    return calibs


def _load_gt_from_file(gt_file_path, index, view_id):
    """Load ground truth annotations from individual file format."""

    annotations = []
    
    if not gt_file_path.exists():
        return annotations
        
    with open(gt_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                annotation = _parse_individual_annotation(line)
                annotations.append(annotation)
            except ValueError as e:
                log.warning(f"Failed to parse annotation in {gt_file_path}: {e}")
                continue
                
    return annotations

def _parse_individual_annotation(line: str) -> dict:
    """Parse a single annotation line from individual format."""
    fields = line.strip().split(',')
    if len(fields) != 9:
        raise ValueError(f"Expected 9 comma-separated values, got {len(fields)}: {line}")

    return {
        "track_id": int(fields[0]),
        "bbox": {
            "x": float(fields[1]),
            "y": float(fields[2]),
            "w": float(fields[3]),
            "h": float(fields[4]),
        },
        "world": {
            "Xw": float(fields[6]),
            "Yw": float(fields[7]),
            "Zw": float(fields[8]),
        }
    }