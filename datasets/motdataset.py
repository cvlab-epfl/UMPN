import json

import numpy as np

from configs.pathes import data_path
from datasets.utils import Calibration, get_frame_from_file
from datasets.sceneset import SceneBaseSet
from misc import geometry
from misc.log_utils import log


class MotDataset(SceneBaseSet):
    def __init__(self, data_conf, dset, sequence, training, dset_type="train"):
        self.root_path = data_path[f'{dset}_root']

        self.dset_name = dset
        self.sequence = sequence

        if dset == "MOT17":
            self.seq_root_path = self.root_path / dset_type / f"{dset}-{sequence}-DPM"
        else:
            self.seq_root_path = self.root_path / dset_type / f"{dset}-{sequence}"

        self.frame_dir_path = self.seq_root_path / "img1"
        self.gt_dir_path = self.seq_root_path / "gt"
        self.gt_reid_path = self.seq_root_path / "annotations_reid/"
        self.gt_det_path = self.seq_root_path / "det" 


        self.gt_features_norm_path = self.seq_root_path / "annotations_features_norm.json"

        self.nb_frames = len([frame_path for frame_path in (self.frame_dir_path).iterdir() if frame_path.suffix == ".jpg"])
        self.calibs = load_calibs_from_json(self.root_path / "calibration_data" / f"{dset}-{sequence}-full_calib.json")

        self.gt = load_gt_mot(self.gt_dir_path / "gt.txt")

        log.debug(f"Leng GT {len(self.gt)}")

        super().__init__(data_conf, training)

        log.debug(f"Dataset {dset} containing {self.nb_frames} frames with views {self.get_views()}")


    def get_frame(self, index, view_id):
            """
            Read and return undistoreted frame coresponding to index and view_id.
            The frame is return at the original resolution
            """

            frame_path = self.frame_dir_path / "{:06d}.jpg".format(index+1)

            # log.debug(f"pomelo dataset get frame {index} {view_id}")
            frame = get_frame_from_file(frame_path)

            return frame

    def _get_gt(self, index, view_id):

        bboxes, person_ids = get_gt_bbox(self.gt, index)

        if len(bboxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 3))
        
        calib = self.get_calib(index)
        
        if len(bboxes) > 0:
            # Get ground points using homography
            world_points = get_ground_point_from_bbox_calib(bboxes, calib)
        else:
            world_points = np.zeros((0, 3))
        
        # Convert person_ids to int type
        person_ids = person_ids.astype(np.int32)
        
        return np.stack(bboxes), np.stack(person_ids), np.stack(world_points)

    def get_det(self, index, view_id):
        
        #call parent function to get det bboxes
        det_bboxes, det_scores = super().get_det(index, view_id)
        calib = self.get_calib(index)

        det_world_points = get_ground_point_from_bbox_calib(det_bboxes, calib)
        
        return det_bboxes, det_scores, det_world_points
        

    
    def get_crops(self, index, view_id, bboxes):
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

    def get_calib(self, index):
        if len(self.calibs) == 1:
            return self.calibs[0]
        else:
            return self.calibs[index]

    def _get_camera_dict(self, index):
        calib = self.get_calib(index)
        camera_position = -calib.R.T @ calib.T
        camera_axis = calib.R.T[:, 2]

        camera_dict = {
            0: {
                'position': camera_position.flatten(),
                'rotation': calib.R,
                'translation': calib.T,
                'intrinsic': calib.K,
                'frame_size': self.frame_sizes[0],
                'axis': camera_axis
            }
        }

        return camera_dict
        

    def get_views(self):
        return [0]
        
    def get_length(self):
        return self.nb_frames

def get_gt_bbox(gt_data, index, class_id=[-1, 1]):
    if len(gt_data) == 0:
        return [], []

    bbox_index_for_frame = np.flatnonzero(gt_data[:,0]==index+1)

    bboxes = []
    person_ids = []

    for bb in bbox_index_for_frame:
        if gt_data[bb,7] in class_id:
            person_id = gt_data[bb,1]
            left = int(gt_data[bb,2]-1)
            top = int(gt_data[bb,3]-1)
            width = int(gt_data[bb,4])
            height = int(gt_data[bb,5])

            # Convert to x1,y1,x2,y2 format
            x1 = left
            y1 = top
            x2 = left + width
            y2 = top + height

            bboxes.append([x1, y1, x2, y2])
            person_ids.append(person_id)

    return np.array(bboxes), np.array(person_ids)

def get_ground_point_from_bbox_calib(bboxes, calib):
    
    bbox_bottom = np.vstack([(bboxes[:,0] + bboxes[:,2]) / 2, bboxes[:, 3]]).T
    ground = geometry.reproject_to_world_ground_batched(bbox_bottom, calib.K, calib.R, calib.T.squeeze())

    return ground


def load_gt_mot(anns_pathes):
    #from mot20 github

    if not anns_pathes.is_file():
        log.warning("NO GT available for this dataset")
        return []
        
    data = np.genfromtxt(anns_pathes, delimiter=',')
    if data.ndim == 1:  # Because in MOT we have different delimiters
        data = np.genfromtxt(anns_pathes, delimiter=' ')
    if data.ndim == 1:  # Because
        print("Ooops, cant parse %s, skipping this one ... " % anns_pathes)

        return None
    # clean nan from results
    #data = data[~np.isnan(data)]
    nan_index = np.sum( np.isnan(data ), axis = 1)
    data = data[nan_index==0]

    return data


def load_calibs_from_json(json_path):
    # Load calibs from JSON file
    with open(json_path, 'r') as f:
        calib_list = json.load(f)
    
    # Convert back to Calibration named tuples
    calibs = []
    for calib_dict in calib_list:
        calib = Calibration(
            K=np.array(calib_dict['K']),
            R=np.array(calib_dict['R']),
            T=np.array(calib_dict['T']), 
            dist=None,
            view_id=calib_dict['view_id']
        )
        calibs.append(calib)
        
    return calibs