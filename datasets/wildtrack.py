import os
import json

import cv2
import numpy as np
import xml.etree.ElementTree as ElementTree

from collections import defaultdict
from pathlib import Path
from xml.dom import minidom

from configs.pathes import data_path
from datasets.utils import Calibration, Bbox, Annotations, get_frame_from_file
from datasets.sceneset import SceneBaseSet
from misc import geometry
from misc.log_utils import log



class WildtrackSet(SceneBaseSet):
    def __init__(self, data_conf, training):
        self.root_path = data_path['wildtrack_root'] 

        self.dset_name = "WILDTRACK"
        self.sequence = 0

        self.frame_dir_path = self.root_path / "Image_subsets/"
        self.gt_dir_path = self.root_path / "annotations_positions/"
        self.gt_reid_path = self.root_path / "annotations_reid/"
        self.gt_det_path = self.root_path / "pretrained_detections/"

        self.gt_features_norm_path = self.root_path / "annotations_features_norm.json"

        self.nb_frames = len([frame_path for frame_path in (self.gt_dir_path).iterdir() if frame_path.suffix == ".json"])
        
        self.calibs = load_calibrations(self.root_path)

        use_det = data_conf["use_detection_training"] if training else data_conf["use_detection_eval"] 
        self.norm_dict_key = f"{'with_det_' + data_conf['detection_type'] if use_det else 'without_det'}"

        super().__init__(data_conf, training)


        log.debug(f"Dataset Wildtrack containing {self.nb_frames} with views {self.get_views()}")


    def get_frame(self, index, view_id):
            """
            Read and return undistoreted frame coresponding to index and view_id.
            The frame is return at the original resolution
            """
            index = index * 5 

            frame_path = self.frame_dir_path / "C{:d}/{:08d}.png".format(view_id + 1, index)

            # log.debug(f"pomelo dataset get frame {index} {view_id}")
            frame = get_frame_from_file(frame_path)

            return frame

    def _get_gt(self, index, view_id):
        index = index * 5 

        gt_path = self.gt_dir_path / "{:08d}.json".format(index)
        annotations_dict = read_json(gt_path, self.calibs)

        view_annotations = annotations_dict[view_id]

        bboxes = []
        person_ids = []
        world_points = []

        for annotation in view_annotations:
            bboxes.append(annotation.bbox)
            person_ids.append(annotation.id)
            world_points.append(annotation.feet_world)
        
        return np.stack(bboxes), np.stack(person_ids), np.stack(world_points)

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
                
                bottom_center = np.array([[xc], [corrected_y2], [1]])
            else:
                bottom_center = np.array([[xc], [y2], [1]])
            
            # Use the calibration for the current view
            calib = self.calibs[view_id]
            K, R, T = calib.K, calib.R, calib.T
            
            # Project bottom center to world coordinates
            world_point = geometry.reproject_to_world_ground(bottom_center, K, R, T, height=0)
            
            det_world_points.append(world_point.squeeze())
        
        det_world_points = np.array(det_world_points)

        return det_bboxes, det_scores, det_world_points
        
        det_world_points = np.array(det_world_points)

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
        return [0,1,2,3,4,5,6]
        
    def get_length(self):
        return self.nb_frames



def load_opencv_xml(filename, element_name, dtype='float32'):
    """
    Loads particular element from a given OpenCV XML file.
    Raises:
        FileNotFoundError: the given file cannot be read/found
        UnicodeDecodeError: if error occurs while decoding the file
    :param filename: [str] name of the OpenCV XML file
    :param element_name: [str] element in the file
    :param dtype: [str] type of element, default: 'float32'
    :return: [numpy.ndarray] the value of the element_name
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError("File %s not found." % filename)
    try:
        tree = ElementTree.parse(filename)
        rows = int(tree.find(element_name).find('rows').text)
        cols = int(tree.find(element_name).find('cols').text)
        return np.fromstring(tree.find(element_name).find('data').text,
                             dtype, count=rows*cols, sep=' ').reshape((rows, cols))
    except Exception as e:
        print(e)
        raise UnicodeDecodeError('Error while decoding file %s.' % filename)
        

def load_all_extrinsics(_lst_files):
    """
    Loads all the extrinsic files, listed in _lst_files.
    Raises:
        FileNotFoundError: see _load_content_lines
        ValueError: see _load_content_lines
    :param _lst_files: [str] path of a file listing all the extrinsic calibration files
    :return: tuple of ([2D array], [2D array]) where the first and the second integers
             are indexing the camera/file and the element of the corresponding vector,
             respectively. E.g. rvec[i][j], refers to the rvec for the i-th camera,
             and the j-th element of it (out of total 3)
    """
#     extrinsic_files = _load_content_lines(_lst_files)
    rvec, tvec = [], []
    for _file in _lst_files:
        xmldoc = minidom.parse(_file)
        rvec.append([float(number)
                     for number in xmldoc.getElementsByTagName('rvec')[0].childNodes[0].nodeValue.strip().split()])
        tvec.append([float(number)
                     for number in xmldoc.getElementsByTagName('tvec')[0].childNodes[0].nodeValue.strip().split()])
    return rvec, tvec


def load_all_intrinsics(_lst_files):

    _cameraMatrices, _distCoeffs = [], []
    for _file in _lst_files:
        _cameraMatrices.append(load_opencv_xml(_file, 'camera_matrix'))
        _distCoeffs.append(load_opencv_xml(_file, 'distortion_coefficients'))
    return _cameraMatrices, _distCoeffs



# Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'dist', 'view_id'])
def load_calibrations(root_path):

    intrinsic_path_format = "calibrations/intrinsic_zero/intr_{}.xml"
    extrinsic_path_format = "calibrations/extrinsic/extr_{}.xml"

    camera_id_to_name = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]

    intrinsic_pathes = [str(root_path / intrinsic_path_format.format(camera)) for camera in camera_id_to_name]
    extrinsic_pathes = [str(root_path / extrinsic_path_format.format(camera)) for camera in camera_id_to_name]

    rotationxyz, T = load_all_extrinsics(extrinsic_pathes)
    K, dist = load_all_intrinsics(intrinsic_pathes)
    
    calib_multi = list()
    for view_id in range(len(intrinsic_pathes)):
#         R = Rotation.from_euler('xyz', rotationxyz[view_id], degrees=False).as_matrix()
        R, _ = cv2.Rodrigues(np.array(rotationxyz[view_id]))

        # dist=dist[view_id]
        calib_multi.append(Calibration(K=K[view_id], R=R, T=np.array(T[view_id])[..., np.newaxis], dist=None, view_id=view_id))

    return calib_multi


# Annotation = namedtuple('Annotation', ['xc', 'yc', 'w', 'h', 'feet', 'head', 'height', 'id', 'frame', 'view'])
def read_json(filename, calib_multi):
    """
    Decodes a JSON file & returns its content.
    Raises:
        FileNotFoundError: file not found
        ValueError: failed to decode the JSON file
        TypeError: the type of decoded content differs from the expected (list of dictionaries)
    :param filename: [str] name of the JSON file
    :return: [list] list of the annotations
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("File %s not found." % filename)
    try:
        with open(filename, 'r') as _f:
            _data = json.load(_f)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode {filename}.")
    if not isinstance(_data, list):
        raise TypeError(f"Decoded content is {type(_data)}. Expected list.")
    if len(_data) > 0 and not isinstance(_data[0], dict):
        raise TypeError(f"Decoded content is {type(_data[0])}. Expected dict.")
        
    multi_view_gt = defaultdict(list)
    
    for person in _data:
        person_id = int(person["personID"])
        frame_id = int(Path(filename).stem)

        for bbox_v in person["views"]:
            if bbox_v["xmax"] == -1:
                continue
            view_id = int(bbox_v["viewNum"])
            xc = (bbox_v["xmax"] + bbox_v["xmin"]) / 2.0
            yc = (bbox_v["ymax"] + bbox_v["ymin"]) / 2.0
            w = bbox_v["xmax"] - bbox_v["xmin"]
            h = bbox_v["ymax"] - bbox_v["ymin"]

            bbox = Bbox(xc=xc, yc=yc, w=w, h=h)
            
            #Compute estimation for position of head and feet
            bbox_bottom_center = np.array([[xc], [h / 2.0 + yc], [1]])
            bbox_top_center = np.array([[xc], [- h / 2.0 + yc], [1]])
            
            calib = calib_multi[view_id]
            K, R, T = calib.K, calib.R, calib.T
            
            #Compute feet and head position in image plane (feet[0]) and in 3d world (feet[1])
            feet_reproj, feet_world  = project_feet(bbox_bottom_center, K, R, T, K, R, T)
            head_reproj, head_world  = project_head(feet_world, bbox_top_center, K, R, T, K, R, T)
            
            height = np.linalg.norm(head_world[1]-feet_world[1])
            
            bbox_corner = (bbox_v["xmin"], bbox_v["ymin"], bbox_v["xmax"], bbox_v["ymax"])
            multi_view_gt[view_id].append(Annotations(bbox=bbox_corner, feet_world=feet_world.squeeze(), head_world=head_world, head=head_reproj,  feet=feet_reproj, height=height, id=person_id, frame=frame_id, view=view_id))
            
    return multi_view_gt


def project_feet(center_bottom, K0, R0, T0, K1, R1, T1):
    
    feet_world = geometry.reproject_to_world_ground(center_bottom, K0, R0, T0)    
    feet_reproj = geometry.project_world_to_camera(feet_world, K1, R1, T1)
    
    return feet_reproj, feet_world
    
def project_head(feet_world, center_top, K0, R0, T0, K1, R1, T1, average_height=165):
    
    head_world = feet_world.copy()
    head_world[2] = average_height

    head_reproj = geometry.project_world_to_camera(head_world, K1, R1, T1)
    
    return head_reproj, head_world