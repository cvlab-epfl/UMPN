from distutils.log import debug
import os
import time

import cv2
import json
import numpy as np
import PIL
import torch 


from collections import namedtuple, defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
from skimage.draw import polygon, polygon_perimeter
from torchvision import transforms

from misc.bbox_utils import calculate_iou
from misc.geometry import  rescale_keypoints, project_image_points_to_groundview
from misc.log_utils import log


#Commonly use namedtuple to encapsulate basic data
Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'dist', 'view_id'])
Bbox = namedtuple('Bbox', ['xc', 'yc', 'w', 'h']) #, 'id', 'frame'])
Annotations = namedtuple('Annotations', ['bbox', 'feet_world', 'head_world', 'head', 'feet', 'height', 'id', 'frame', 'view'])
Homography = namedtuple('Homography', ['H', 'input_size', 'output_size'])


def get_augmentation(tuple_aug_list, frame_size):

    aug_dir = {
        "rcrop": transforms.RandomResizedCrop(frame_size),
        "raff": transforms.RandomAffine(degrees = 45, translate = (0.2, 0.2), scale = (0.8,1.2), shear = 10),
        "raff_nor" : transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2), scale = (0.8,1.2), shear = 10),
        "raff_hr" : transforms.RandomAffine(degrees = 360, translate = (0.2, 0.2), scale = (0.8,1.2), shear = 10),
        "raff_nos" : transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2), scale = None, shear = None),
        "raff_ht" : transforms.RandomAffine(degrees = 0, translate = (0.4, 0.4), scale = None, shear = None),
        "vflip"  : transforms.RandomVerticalFlip(p = 1.0),
        "hflip"  : transforms.RandomHorizontalFlip(p = 1.0),
        "rpersp" : transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
        "rpersph" : transforms.RandomPerspective(distortion_scale=0.3, p=1.0),
        "rpersps" : transforms.RandomPerspective(distortion_scale=0.7, p=1.0),
        None:None   
    }

    tuple_aug = [(aug_dir[augname], prob) for augname, prob in tuple_aug_list]

    #Making sure probability of augmentation sums up to one
    total_prob = sum([prob for augname, prob in tuple_aug])
    if total_prob > 1:
        log.warning(f"Augmentation probability sums up to {total_prob}, rescaling such that it sums up to one")
        tuple_aug = [(augname, prob/total_prob) for augname, prob in tuple_aug_list]
    elif total_prob < 1:
        tuple_aug.append((None, 1-total_prob))

    return tuple_aug

def is_in_frame(point, frame_size):
    is_in_top_left = point[0] > 0 and point[1] > 0
    is_in_bottom_right = point[0] < frame_size[0] and point[1] < frame_size[1]
    
    return is_in_top_left and is_in_bottom_right


def check_if_in_roi(roi, point):
    #if point is completely ou of roi return false
    if point[0] >= roi.shape[-1] or point[1] >= roi.shape[-2] or point[0] < 0 or point[1] < 0:
        return False
    
    return roi[point[1], point[0]] == 1

def find_nearest_point(point, point_list):
    
    point_list = np.asarray(point_list)
    dist_2 = np.sum((point_list - point)**2, axis=1)
    
    return np.argmin(dist_2)

def interpolate_annotations(ann_dict, index):
    
    existing_index = sorted(list(ann_dict.keys()))
    
    ind_pos = np.searchsorted(existing_index, index)

    if ind_pos >= len(existing_index):
        # log.warning("Try to interpolate with no following annotation")
        return []

    before = existing_index[ind_pos - 1]
    after = existing_index[ind_pos]
    
    inter_weight = (index - before) / (after - before)
    
    ann_interpolated = list()

    for ann in ann_dict[before]:
        ann_after = [ann_after for ann_after in ann_dict[after] if ann_after.id == ann.id]
        if len(ann_after) > 0:
            assert len(ann_after) == 1

            ann_after = ann_after[0]
            
            if ann.head is not None and ann_after.head is not None:
                head_reproj = ann.head*(1-inter_weight) + ann_after.head*inter_weight
            else:
                head_reproj = None
            
            if ann.height is not None and ann_after.height is not None:
                height = ann.height*(1-inter_weight) + ann_after.height*inter_weight
            else:
                height = None

            #is feet is not visible in frame before or after we remove annotation
            if ann.feet is not None and ann_after.feet is not None:  
                feet_reproj = ann.feet *(1-inter_weight)+ ann_after.feet*inter_weight

                ann_interpolated.append(Annotations(bbox=None, head=head_reproj,  feet=feet_reproj, height=height, id=ann.id, frame=index, view=ann.view))

    return ann_interpolated

def undistort_gt(ann_list, K, dist_coeff):
    undi_anns = list()
    for ann in ann_list:
        
        if ann.head is not None:
            head_undi = cv2.undistortPoints(ann.head, K, dist_coeff, P=K).reshape(2,1)
        else:
            head_undi = ann.head

        if ann.feet is not None:
            feet_undi = cv2.undistortPoints(ann.feet, K, dist_coeff, P=K).reshape(2,1)
        else:
            feet_undi = ann.feet
        
        undi_anns.append(ann._replace(head=head_undi, feet=feet_undi))

    return undi_anns


def resize_density(hm, hm_size, scale_factor=None, mode="bilinear"):
    '''
    Resize heatmap and rescale value in order to keep density consitent    
    '''
    
    if np.all(np.array(hm.shape[-2:]) == np.array(hm_size)):
        # Size already correct
        return hm

    hm_resized = torch.nn.functional.interpolate(hm, size=tuple(hm_size), mode=mode)

    if scale_factor is None:
        scale_factor = hm.sum() / hm_resized.sum()
        # scale_factor = np.prod(np.array(hm.shape[-2:]) / np.array(hm_size))

    #Scale heatmap such that density sum is equal after resizing
    hm_resized = hm_resized * scale_factor

    return hm_resized


def read_json_file(filepath):
    with open(os.path.abspath(filepath)) as f:    
        json_dict = json.load(f)

    return json_dict

def write_dict_as_json(data_dict, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(data_dict, outfile, ensure_ascii=False, indent=4)

def read_sloth_annotation(ann_pathes):
    
    scene_ann = []
    
    person_id_set = set()
    for view_id, ann_view in enumerate(ann_pathes):
        
        view_ann = dict()
        
        data = read_json_file(ann_view)
        
        print(f"Scene {view_id+1} containing {len(data)} frames")
        
        for frame_json in data:
            frame_id = int(frame_json["filename"].split("/")[-1][12:-4])
            frame_ann = defaultdict(dict)
            
            for point_ann in frame_json["annotations"]:
                person_id_set.add(int(point_ann["person_id"]))
                frame_ann[int(point_ann["person_id"])][point_ann["class"]] =  (point_ann["x"], point_ann["y"])
            
            view_ann[frame_id] = frame_ann
            
        scene_ann.append(view_ann)
    
    return scene_ann, list(person_id_set)


def get_frame_from_file(frame_path):
    
    if not(frame_path.is_file()):
        log.error(f"Trying to load {frame_path} which is not a file")
        assert frame_path.is_file()

    frame = cv2.imread(str(frame_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame  


def get_frame_from_video(video_path, frame_index):
    while True:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        res, frame = cap.read()
        
        # log.debug(f"reading frame from video {res}{frame.shape}")
        cap.release()
        
        if res and frame is not None:
            break

        time.sleep(1)

    return frame   


def get_train_val_split(scene_set_cumsum_lengths, split_proportion):
    train_idx = list()
    test_idx = list()

    for i in range(len(scene_set_cumsum_lengths)):
        set_end = scene_set_cumsum_lengths[i]

        if i != 0:
            prev_end = scene_set_cumsum_lengths[i-1]
        else:
            prev_end = 0
        
        train_size = int((set_end - prev_end) * split_proportion)

        train_idx.extend(list(range(prev_end, prev_end + train_size)))
        test_idx.extend(list(range(prev_end + train_size, set_end)))
    
    return train_idx, test_idx

def get_train_val_split_index(dataset, dataset_val, split_proportion):
    train_size = int(len(dataset) * split_proportion)
    val_size = len(dataset) - train_size

    train_idx = list(range(0, train_size))
    test_idx = list(range(len(dataset_val) - val_size, len(dataset_val)))

    return train_idx, test_idx


def generate_scene_roi_from_view_rois(view_ROIs, view_homographies, frame_original_size, homography_input_size, homography_output_size, hm_size):
    """
    given region of interest (defines as points on the perimeter of the the region) generate 
    the region of interest and it's boundary for the full scene in the grounplane caracterised by the homographies.
    """

    #rescame0 ROI corner point in image to match homography size
    if frame_original_size != homography_input_size:
        view_ROIs = [rescale_keypoints(roi_v, frame_original_size, homography_input_size)  for roi_v in view_ROIs]

    view_ROIs_ground = [project_image_points_to_groundview(img_roi, homography) for img_roi, homography in zip(view_ROIs, view_homographies)]

    #rescale ROI corner in grounplane to match output hm size
    if homography_output_size != hm_size:
        view_ROIs_ground =  [rescale_keypoints(roi_ground, homography_output_size, hm_size)  for roi_ground in view_ROIs_ground]

    #merge the polygons from each view ROI into a single polygon
    polygons = [Polygon(roi_ground) for roi_ground in view_ROIs_ground]

    # polygons = [MultiPoint(roi_ground).convex_hull for roi_ground in view_ROIs_ground]

    try:
        # print("--------------------------")
        # for poy in polygons:
        #     print(list(poy.exterior.coords))
        # print(unary_union(polygons))
        # print("########################")

        ROI_ground_poly = np.array(list(unary_union(polygons).exterior.coords))
    except:
        log.warning("ROI couldn't be generated, returning ROI covering the full groundplane")
        return np.ones(hm_size), np.zeros(hm_size)

    roi = np.zeros(hm_size)
    mask_boundary = np.zeros(hm_size)

    rr, cc = polygon_perimeter(ROI_ground_poly[:,1], ROI_ground_poly[:,0], mask_boundary.shape, clip=True)
    mask_boundary[rr,cc] = 1
    mask_boundary = mask_boundary

    roi[rr,cc] = 1
    rr, cc = polygon(ROI_ground_poly[:,1], ROI_ground_poly[:,0], roi.shape)
    roi[rr,cc] = 1

    return roi, mask_boundary

def generate_mask_from_polygon_perimeter(perimeter, image_size):
    image = np.zeros(image_size)
    rr, cc = polygon(perimeter[:,1], perimeter[:,0], image.shape)
    image[rr,cc] = 1

    return image


def aggregate_multi_view_gt(anns_gt, homographies, frame_original_size, homography_input_size, homography_output_size, hm_size):
        feets_ground = defaultdict(lambda: defaultdict(list))

        #Apply groundplane homography to all annotation from differents view
        for view_id in range(len(homographies)):
            anns_view = anns_gt[view_id]
            h_view = homographies[view_id]

            for ann in anns_view:
                
                feet_loc = ann.feet.reshape(1,2)
                
                # log.debug(f"feet_loc {feet_loc.shape}")

                #rescale input point if they don't match homography input size
                if frame_original_size != homography_input_size:
                    feet_loc =  rescale_keypoints(feet_loc, frame_original_size, homography_input_size)
                
                # log.debug(f"feet_loc r {feet_loc.shape}")

                feet_ground = project_image_points_to_groundview(feet_loc, h_view)

                # log.debug(f"feet_ground {feet_ground.shape}")

                #rescale feet point in the ground according to final heatmap dimension
                if homography_output_size != hm_size:
                    feet_ground =  rescale_keypoints(feet_ground, homography_output_size, hm_size)

                # log.debug(f"feet_ground r {feet_ground.shape}")

                feets_ground[ann.frame][ann.id].append(feet_ground.squeeze())

        #Aggregate annotation and generate final groundplane annoatation
        anns_ground = list()
        for frame_id, frame_anns in feets_ground.items():
            for person_id, feet_ground_list in frame_anns.items():
                # log.debug(f"feet_ground-list shape {np.array(feet_ground_list).shape}")
                # log.debug(f"feet_ground-list {np.array(feet_ground_list)}")
                # log.debug(f"feet_ground-list mean {np.mean(np.array(feet_ground_list), axis=0)}")

                feet_ground_agg = np.mean(np.array(feet_ground_list), axis=0)
                anns_ground.append(Annotations(bbox=None, head=None,  feet=feet_ground_agg, height=None, id=person_id, frame=frame_id, view="ground"))

        return anns_ground


def extract_points_from_gt(gt, hm_size, gt_original_size=None, get_mask=False):
        '''
        extract point and person id from a list of gt named tuple
        '''

        if len(gt) == 0:
            if get_mask:
                return gt, [], []
            else:
                return gt, []

        #Filter visible point in that view
        gt_points = np.array([ann.feet for ann in gt if ann.feet is not None])
        person_id = np.array([ann.id for ann in gt if ann.feet is not None])

        if len(gt_points.shape) > 2:
            gt_points = gt_points.squeeze(2)

        if gt_original_size is not None and gt_original_size != hm_size:
            assert len(gt_points.shape) <= 2, f"gt points should have two dimensions: {gt_points.shape}"
            gt_points =  rescale_keypoints(gt_points, gt_original_size, hm_size)

        # log.debug(gt_points.shape)
        # log.debug(person_id.shape)

        # log.debug(hm_size)

        #Filter out point outside of hm size
        mask_visible = np.array([is_in_frame(np.rint(point[[1,0]]), hm_size) for point in gt_points])
        gt_points = gt_points[mask_visible] 
        person_id = person_id[mask_visible]

        if get_mask:
            return gt_points, person_id, mask_visible
        else:
            return gt_points, person_id

def aggregate_multi_view_gt_points(gt_points, gt_person_id, homographies, frame_original_size, homography_input_size, homography_output_size, hm_size):
        """
        Takes a list of list of points and person id, and a list of homographies.
        Project the list of each view in a common groundplane using homographies, and aggregate projection to generate gt in grounplane
        """
        points_ground = defaultdict(list)

        #Apply groundplane homography to all annotation from differents view
        for view_id in range(len(homographies)):
            gt_view = gt_points[view_id]
            id_view = gt_person_id[view_id]
            h_view = homographies[view_id]

            for point, person_id in zip(gt_view, id_view):
                
                point = point.reshape(1,2)

                #rescale input point if they don't match homography input size
                if frame_original_size != homography_input_size:
                    point =  rescale_keypoints(point, frame_original_size, homography_input_size)
                
                # log.debug(f"feet_loc r {feet_loc.shape}")

                point_ground = project_image_points_to_groundview(point, h_view)

                # log.debug(f"feet_ground {feet_ground.shape}")

                #rescale feet point in the ground according to final heatmap dimension
                if homography_output_size != hm_size:
                    point_ground =  rescale_keypoints(point_ground, homography_output_size, hm_size)

                # log.debug(f"feet_ground r {feet_ground.shape}")

                points_ground[person_id].append(point_ground.squeeze())

        #Aggregate annotation and generate final groundplane annoatation
        agg_points_ground = list()
        for person_id, feet_ground_list in points_ground.items():
            # log.debug(f"feet_ground-list shape {np.array(feet_ground_list).shape}")
            # log.debug(f"feet_ground-list {np.array(feet_ground_list)}")
            # log.debug(f"feet_ground-list mean {np.mean(np.array(feet_ground_list), axis=0)}")

            feet_ground_agg = np.mean(np.array(feet_ground_list), axis=0)

            if is_in_frame(np.rint(feet_ground_agg[[1,0]]), hm_size):
                agg_points_ground.append((feet_ground_agg, person_id))

        if len(agg_points_ground) == 0:
            return [], []
            
        gt_points_ground, gt_ground_person_id = [np.array(x) for x in zip(*agg_points_ground)]

        return gt_points_ground, gt_ground_person_id

def get_flow_channel(x, y, x_prev, y_prev):

    #deal with radius larger than one:
    if x_prev > x:
        x_prev = x + 1
    elif x_prev < x:
        x_prev = x - 1

    if y_prev > y:
        y_prev = y + 1
    elif y_prev < y:
        y_prev = y - 1

    if x == x_prev and y == y_prev:
        return 4
    if x == x_prev and y == y_prev+1:
        return 7
    if x == x_prev+1 and y == y_prev:
        return 5
    if x == x_prev+1 and y == y_prev+1:
        return 8
    if x == x_prev and y == y_prev-1:
        return 1
    if x == x_prev-1 and y == y_prev:
        return 3
    if x == x_prev-1 and y == y_prev-1:
        return 0
    if x == x_prev+1 and y == y_prev-1:
        return 2
    if x == x_prev-1 and y == y_prev+1:
        return 6

def generate_motion_tuple(pre_gt, pre_gt_person_id, gt, gt_person_id):
    
    if len(pre_gt) != 0:
        pre_gt = np.rint(pre_gt).astype(int)

    if len(gt) != 0:
        gt = np.rint(gt).astype(int)
    
    if not isinstance(pre_gt_person_id, list):
        pre_gt_person_id = pre_gt_person_id.tolist()
    if not isinstance(gt_person_id, list):
        gt_person_id = gt_person_id.tolist()

    # only keep person id present in both timestep
    common_person_id = set(pre_gt_person_id).intersection(gt_person_id)

    #create tuple of positions for each person present both in pre_gt and gt
    common_position = [(pre_gt[pre_gt_person_id.index(p_id)], gt[gt_person_id.index(p_id)]) for p_id in common_person_id]        
    
    return common_position
    
def generate_flow(pre_gt, pre_gt_person_id, gt, gt_person_id, roi, hm_radius, generate_hm=True):
    
    if generate_hm:
        gt_flow = np.zeros((10, roi.shape[-2], roi.shape[-1]))
    else:
        gt_flow = None

    roi = roi.squeeze()

    common_position = generate_motion_tuple(pre_gt, pre_gt_person_id, gt, gt_person_id) 
    
    gt_flow_tuple = list()
    move_length = list()
    for (pres_pos, pos) in common_position:
        nb_pixel_move = cdist([pres_pos], [pos], 'chebyshev').item()
        move_length.append(nb_pixel_move)
        if  nb_pixel_move <= 1:
            if check_if_in_roi(roi, pres_pos):
                flow_channel = get_flow_channel(pos[0], pos[1], pres_pos[0], pres_pos[1])
                # log.debug(f"{pres_pos}, {pos}, {flow_channel}")
                if generate_hm:
                    cv2.circle(gt_flow[flow_channel], tuple(pres_pos), hm_radius, 1, thickness=cv2.FILLED)
                gt_flow_tuple.append((pres_pos, flow_channel))
            elif check_if_in_roi(roi, pos):
                #incoming flow from the outside of roi
                if generate_hm:
                    cv2.circle(gt_flow[9], tuple(pos), hm_radius, 1, thickness=cv2.FILLED)
                gt_flow_tuple.append((pos, 9))

    if generate_hm:
        gt_flow = torch.from_numpy(gt_flow).to(torch.float32)
        
    
    return gt_flow, gt_flow_tuple, move_length


def find_nearest_edge(point, frame_size):
    x, y = point[0], point[1]
    
    closest_edge = np.argmin([np.abs(x-frame_size[1]), np.abs(x), np.abs(y-frame_size[0]), np.abs(y)])
    
    if closest_edge == 0:
        return np.array([frame_size[1], y])
    elif closest_edge == 1:
        return np.array([0, y])
    elif closest_edge == 2:
        return np.array([x, frame_size[0]])
    elif closest_edge == 3:
        return np.array([x, 0])


def set_value_for_square(array, center_point, value, radius, frame_size):
    x, y = center_point[0], center_point[1]
    height, width = frame_size

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    array[:, y - top:y + bottom, x - left:x + right] = value
#     tracking_mask[:, y - top:y + bottom, x - left:x + right] = 1
    
    return array


def build_tracking_gt(pre_person_id, person_id, pre_gt_points, gt_points, hm_size, gt_radius, exiting=False, entering=False):
    intersect_id, ind, pre_ind = np.intersect1d(person_id, pre_person_id, return_indices=True)

    #Initialize map and mask
    tracking_map = np.zeros((2, hm_size[0], hm_size[1]))
    tracking_mask = np.zeros((1, hm_size[0], hm_size[1]))
    exiting_mask = np.zeros((1, hm_size[0], hm_size[1]))
    entering_mask = np.zeros((1, hm_size[0], hm_size[1]))
    

    for i in range(intersect_id.shape[0]):
        frame_id = ind[i]
        pre_id = pre_ind[i]

        point = np.rint(pre_gt_points[pre_id]).astype(int)
        offset = (gt_points[frame_id] - pre_gt_points[pre_id])[..., np.newaxis, np.newaxis]
        
        tracking_map = set_value_for_square(tracking_map, point, offset, gt_radius, hm_size)
        tracking_mask = set_value_for_square(tracking_mask, point, np.ones((1,1)), gt_radius, hm_size)
        
    
    if exiting:
        #Dealing with people exiting the scene
        exiting_inds = np.where(np.invert(np.isin(pre_person_id, person_id)))[0]

        for ex_ind in exiting_inds:
            point = np.rint(pre_gt_points[ex_ind]).astype(int)
            exit_point = find_nearest_edge(point, hm_size)
            
            offset = (exit_point - point)[..., np.newaxis, np.newaxis]

            tracking_map = set_value_for_square(tracking_map, point, offset, gt_radius, hm_size)
            exiting_mask = set_value_for_square(exiting_mask, point, np.ones((1,1)), gt_radius, hm_size)
    
    if entering:
        #Dealing with people entering the scene
        entering_inds = np.where(np.invert(np.isin(person_id, pre_person_id)))[0]    
        for ent_ind in entering_inds:
            point = np.rint(gt_points[ent_ind]).astype(int)
            enter_point = find_nearest_edge(point, hm_size)
            
            offset = (point - enter_point)[..., np.newaxis, np.newaxis]
            
            tracking_map = set_value_for_square(tracking_map, enter_point, offset, gt_radius, hm_size)
            entering_mask = set_value_for_square(entering_mask, enter_point, np.ones((1,1)), gt_radius, hm_size)

    return tracking_map, tracking_mask, exiting_mask, entering_mask


def merge_gt_with_det(bbox, world_points, label, person_id, bbox_det, bbox_scores, world_points_det, iou_threshold=0.1, include_unmatched_gt=True, unmatched_gt_confidence=(0.3,1)):

    # Compute IoU matrix
    iou_matrix = np.zeros((len(bbox), len(bbox_det)))
    for i, gt_box in enumerate(bbox):
        for j, det_box in enumerate(bbox_det):
            iou_matrix[i, j] = calculate_iou(gt_box, det_box)

    # Apply Hungarian algorithm for optimal matching
    gt_indices, det_indices = linear_sum_assignment(-iou_matrix)

    final_bbox = []
    final_label = []
    final_confidence = []
    final_person_id = []
    final_world_points = []

    matched_gt = set()
    matched_det = set()

    # Process matched boxes
    for gt_idx, det_idx in zip(gt_indices, det_indices):
        if iou_matrix[gt_idx, det_idx] >= iou_threshold:
            final_bbox.append(bbox_det[det_idx])
            final_label.append(1)
            final_confidence.append(bbox_scores[det_idx][0])
            final_person_id.append(person_id[gt_idx])
            final_world_points.append(world_points_det[det_idx])
            matched_gt.add(gt_idx)
            matched_det.add(det_idx)

    # Add unmatched GT boxes if flag is set
    if include_unmatched_gt:
        for i in range(len(bbox)):
            if i not in matched_gt:
                final_bbox.append(bbox[i])
                final_label.append(1)
                final_confidence.append(np.random.uniform(unmatched_gt_confidence[0], unmatched_gt_confidence[1]))
                final_person_id.append(person_id[i])
                final_world_points.append(world_points[i])

    # Add unmatched detection boxes
    for i in range(len(bbox_det)):
        if i not in matched_det:
            final_bbox.append(bbox_det[i])
            final_label.append(0)
            final_confidence.append(bbox_scores[i][0])
            final_person_id.append(-1)  # Use -1 for unmatched detections
            final_world_points.append(world_points_det[i])

    return np.array(final_bbox, dtype=np.int32), np.array(final_label, dtype=np.int32), np.array(final_confidence, dtype=np.float32), np.array(final_person_id, dtype=np.int32), np.array(final_world_points, dtype=np.float32)


def process_crops(crops, size=(256, 80)):
    """
    Process the list of crops by resizing and normalizing them.

    Args:
        crops (list): List of crop images.
        size (tuple): Desired size for the crops (height, width).

    Returns:
        torch.Tensor: Tensor containing the processed crops.
    """
    if len(crops) == 0:
        return torch.empty((0, 3, size[0], size[1]))

    processed_crops = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for crop in crops:
        if crop is not None:
            processed_crop = transform(crop)
            processed_crops.append(processed_crop)
        else:
            # If crop is None, create a tensor of zeros with the desired size
            processed_crops.append(torch.zeros((3, size[0], size[1])))

    return torch.stack(processed_crops)