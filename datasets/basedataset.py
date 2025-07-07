import numpy as np
import torch

from misc.utils import PinnableDict
from misc.log_utils import log


class MvSequenceSet(torch.utils.data.Dataset):
    
    def __init__(self, scene_set, data_conf):
    
        #nb_view, size, hm_builder, hm_size, frame_interval, hm_radius, pre_range, pre_hm_fp_rate, pre_hm_fn_rate, use_aug_train, image_plane, ground_input, multiview_as_multiset):
        
        """
        nb_view : number of view the dataset will return
        size : output size of image and heatmpas
        hm_builder : function use to make heatmap 
        hm_radius : radius of gaussian use when building heatmap
        pre_range : distance to select previous frame
        """
        self.scene_set = scene_set
        self.view_names = scene_set.get_views()

        self.features = data_conf["features"]

        self.use_augmentation = data_conf["aug_train"] and self.scene_set.training

        log.debug(f"Flow scene set containing {len(self.view_names)} views and {len(self.scene_set)} frames, use_augmentation: {self.use_augmentation}")

        # self.use_augmentation = data_conf["aug_train"] and training

        # #list of augmentation and their probability
        # self.view_based_augs = [(HomographyDataAugmentation(aug), prob) if aug is not None else (None, prob) for aug, prob in get_augmentation(data_conf["views_based_aug_list"], self.frame_input_size)]
        # self.scene_based_augs = [(HomographyDataAugmentation(aug), prob) if aug is not None else (None, prob) for aug, prob in get_augmentation(data_conf["scene_based_aug_list"], self.hm_size)]

        # if self.use_augmentation:
        #     log.info(f"View base augmentation {self.view_based_augs}")
        #     log.info(f"Scene base augmentation {self.scene_based_augs}")


        # if sum([x[1] for x in self.scene_based_augs]) != 1:
        #     log.warning(f"Scene based augmentation probability should sum up to one but is {sum([x[1] for x in self.scene_based_augs])}")
        # if sum([x[1] for x in self.view_based_augs]) != 1:
        #     log.warning(f"View based augmentation probability should sum up to one but is {sum([x[1] for x in self.view_based_augs])}")


    def __getitem__(self, index):
        multi_view_data = dict()
        
        # scene_based_aug = self.select_augmentation(*zip(*self.scene_based_augs))

        for view_id in self.view_names:
            frame_dict = {}
            # frame_dict["view_id"] = view_id

            # view_based_aug = self.select_augmentation(*zip(*self.view_based_augs))

            #bbox, world_points, person_id, reid, crops 
            view_dict = self.scene_set.get(index, view_id)

            if self.use_augmentation:
                view_dict = self.aug_view_dict(view_dict)            

            # normalized_det_representation, feature_map = self.normalize_det_representation(view_dict, normalization_dict)
            # frame, homography, gt_points_image, person_id_image = self.apply_view_based_augmentation(frame, homography, gt_points_image, person_id_image, view_based_aug)
            # gt_points_image = np.rint(gt_points_image)

            frame_dict.update(view_dict)
            # frame_dict["feature_map"] = feature_map
            # frame_dict["normalized_det_representation"] = normalized_det_representation
            multi_view_data[view_id] = frame_dict
        
        normalization_dict = self.scene_set.get_norm_dict()
        multi_view_data["normalization_dict"] = normalization_dict

        camera_dict = self.scene_set.get_camera_dict(index)
        multi_view_data["camera_dict"] = camera_dict

        # log.debug(multi_view_data)
        # multi_view_data = listdict_to_dictlist(multi_view_data)
        # multi_view_data = stack_tensors(multi_view_data)

        return multi_view_data

    def aug_view_dict(self, view_dict, bbox_jitter_prob=0.5, bbox_jitter_scale=0.05, 
                      bbox_scale_prob=0.3, bbox_scale_range=0.1, 
                      conf_noise_prob=0.3, conf_noise_scale=0.3):
        """
        Apply standard augmentations to bounding boxes and confidence scores.
        
        Args:
            view_dict: Dictionary containing detection data
            bbox_jitter_prob: Probability of applying jitter to bounding boxes
            bbox_jitter_scale: Scale of jitter relative to bbox dimensions
            bbox_scale_prob: Probability of applying scaling to bounding boxes
            bbox_scale_range: Range of scaling factor (±value)
            conf_noise_prob: Probability of adding noise to confidence scores
            conf_noise_scale: Scale of noise added to confidence (±value)
            
        Returns:
            Augmented view_dict
        """

        if "bbox" in view_dict and len(view_dict["bbox"]) > 0:
            # Get original values
            bbox = view_dict["bbox"].clone() if isinstance(view_dict["bbox"], torch.Tensor) else torch.tensor(view_dict["bbox"], dtype=torch.float32)
            confidence = view_dict["confidence"].clone() if isinstance(view_dict["confidence"], torch.Tensor) else torch.tensor(view_dict["confidence"], dtype=torch.float32)
            
            # Random jitter to bounding boxes (small perturbations)
            if np.random.random() < bbox_jitter_prob:
                # Apply random jitter to each coordinate (x1, y1, x2, y2)
                width = bbox[:, 2] - bbox[:, 0]
                height = bbox[:, 3] - bbox[:, 1]
                
                jitter_x = width * bbox_jitter_scale * (torch.rand_like(width) - 0.5) * 2
                jitter_y = height * bbox_jitter_scale * (torch.rand_like(height) - 0.5) * 2
                
                # Apply jitter while maintaining box structure (x1 < x2, y1 < y2)
                bbox[:, 0] = bbox[:, 0] + jitter_x
                bbox[:, 1] = bbox[:, 1] + jitter_y
                bbox[:, 2] = bbox[:, 2] + jitter_x
                bbox[:, 3] = bbox[:, 3] + jitter_y
            
            # Random scaling of bounding boxes
            if np.random.random() < bbox_scale_prob:
                scale_factor = 1.0 + (torch.rand(bbox.size(0)) * 2 * bbox_scale_range - bbox_scale_range)
                center_x = (bbox[:, 0] + bbox[:, 2]) / 2
                center_y = (bbox[:, 1] + bbox[:, 3]) / 2
                width = (bbox[:, 2] - bbox[:, 0]) * scale_factor
                height = (bbox[:, 3] - bbox[:, 1]) * scale_factor
                
                bbox[:, 0] = center_x - width / 2
                bbox[:, 1] = center_y - height / 2
                bbox[:, 2] = center_x + width / 2
                bbox[:, 3] = center_y + height / 2
            
            # Random adjustment to confidence scores
            if np.random.random() < conf_noise_prob:
                confidence_noise = conf_noise_scale * (torch.rand_like(confidence) - 0.5) * 2
                confidence = torch.clamp(confidence + confidence_noise, 0.0, 1.0)
            
            # Update the view_dict with augmented values
            view_dict["bbox"] = bbox
            view_dict["confidence"] = confidence

        return view_dict

    def normalize_det_representation(self, view_dict, normalization_dict):
        normalized_features = []
        feature_map = {}
        current_index = 0

        for feature in self.features:
            feature_data = view_dict[feature]
            mean = normalization_dict[feature]["mean"]
            std = normalization_dict[feature]["std"]
            
            normalized = (feature_data - mean) / std
            feature_length = len(normalized)
            
            normalized_features.append(normalized)
            feature_map[feature] = (current_index, current_index + feature_length)
            current_index += feature_length

        normalized_vector = np.concatenate(normalized_features)
        
        return normalized_vector, feature_map

    
    def disturb_points(self, gt_points):
        
        disturbed_points = list()

        #Leave pre hm empty sometime
        if np.random.random() > 0.1:
            for point in gt_points:

                #Drop gt point to generate False negative
                if np.random.random() > self.pre_hm_fp_rate:
                    disturbed_points.append(point)

                #Add False positive near existing point
                if np.random.random() < self.pre_hm_fn_rate:
                    point_fn = point.copy()
                    point_fn[0] = point_fn[0] + np.random.randn() * 0.05 * self.hm_size[0]
                    point_fn[1] = point_fn[1] + np.random.randn() * 0.05 * self.hm_size[1]
                    
                    disturbed_points.append(point_fn)

        disturbed_points = np.array(disturbed_points)

        return disturbed_points

   
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __len__(self):
        return len(self.scene_set)

    @staticmethod
    def collate_fn(batch):
        
        assert len(batch) == 1, "Batch size should be 1"
        #Flatten list of list
        # batch = [item for sublist in batch for item in sublist]
        # if len(batch) == 1:
        #     expand_dim = True

        #Merge dictionnary
        # batch = listdict_to_dictlist(batch)
        # batch = stack_tensors(batch)

        # log.debug(batch["frame"].shape)
        # if expand_dim:
        #     #when batchsize is one we add an empty batch dimension
        #     batch = expand_tensors_dim(batch, 0)
        # log.debug(batch["frame"].shape)

        collate_dict = PinnableDict(batch[0])
        # log.info(collate_dict["frame"].shape)

        # log.spam(f"collate_dict {type(collate_dict)}")

        return collate_dict