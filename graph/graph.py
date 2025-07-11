import torch
import numpy as np
import json

from torch_geometric.data import HeteroData

from collections import OrderedDict
from graph import postprocessing
from graph import visualization
from graph.utils import get_graph_stats, compute_graph_cost_from_id
from misc.bbox_utils import calculate_bbox_iou, calculate_bbox_ratio
from misc.log_utils import log
from misc.metrics import calculate_classification_metrics, get_tracking_metrics

class HeteroGraph:
    """
    This class is used to store the graph data for the model.

    The structure of self.data:
    - Node types:
        - 'detection': Represents individual detections
        - 'camera': Represents camera information
    
    - Edge types:
        - ('detection', 'temporal', 'detection'): Temporal edges between detections
        - ('detection', 'view', 'detection'): View edges between detections
        - ('detection', 'social', 'detection'): Social edges between detections
        - ('camera', 'next', 'camera'): Next edges between cameras
        - ('camera', 'see', 'detection'): See edges from cameras to detections
    
    - Node features and metadata:
        - self.data['detection'].x: Node features for detections [num_detections, num_features]
        - self.data['camera'].x: Node features for cameras [num_cameras, num_camera_features]
        - self.data['detection'].label: Labels for detection nodes [num_detections]
        - self.data['detection'].pred: Predictions for detection nodes [num_detections]
        - self.data['detection'].timestamp: Timestamp information for detections [num_detections]
        - self.data['detection'].view_id: View ID for each detection [num_detections]
        - self.data['detection'].person_id: Person ID for each detection [num_detections]
        - self.data['detection'].bbox: Bounding box information for detections [num_detections, 4]
        - self.data['detection'].world_points: World coordinates for detections [num_detections, 3]
    
    - Edge indices:
        - self.data['detection', 'temporal', 'detection'].edge_index: Indices for temporal edges [2, num_temporal_edges]
        - self.data['detection', 'view', 'detection'].edge_index: Indices for view edges [2, num_view_edges]
        - self.data['detection', 'social', 'detection'].edge_index: Indices for social edges [2, num_social_edges]
        - self.data['camera', 'next', 'camera'].edge_index: Indices for next edges [2, num_next_edges]
        - self.data['camera', 'see', 'detection'].edge_index: Indices for see edges [2, num_see_edges]
    
    - Edge features:
        - self.data['detection', 'temporal', 'detection'].edge_attr: Features for temporal edges [num_temporal_edges, num_edge_features]
        - self.data['detection', 'view', 'detection'].edge_attr: Features for view edges [num_view_edges, num_edge_features]
        - self.data['detection', 'social', 'detection'].edge_attr: Features for social edges [num_social_edges, num_edge_features]
        - self.data['camera', 'next', 'camera'].edge_attr: Features for next edges [num_next_edges, num_edge_features]
        - self.data['camera', 'see', 'detection'].edge_attr: Features for see edges [num_see_edges, num_edge_features]
    
    - Edge labels and predictions:
        - self.data['detection', 'temporal', 'detection'].edge_label: Labels for temporal edges [num_temporal_edges]
        - self.data['detection', 'temporal', 'detection'].edge_pred: Predictions for temporal edges [num_temporal_edges]
        - self.data['detection', 'view', 'detection'].edge_label: Labels for view edges [num_view_edges]
        - self.data['detection', 'view', 'detection'].edge_pred: Predictions for view edges [num_view_edges]

    This structure allows for efficient storage and manipulation of heterogeneous graph data,
    including multiple node types, edge types, and their associated features and metadata.
    The dimensionality of each attribute is provided to give a clear understanding of the data structure.

    """

    def __init__(self, model_conf, data_conf, training=False, device="cpu"):
        self.device = device
        
        self.data = HeteroData()
        self.archived_data = HeteroData()
        self.nb_timesteps = 0

        self.features = data_conf["features"]

        self.edge_features = {
            "temporal": model_conf["temporal_edges_features"],
            "view": model_conf["view_edges_features"],
            "social": model_conf["social_edges_features"]
        }

        self.window_size = model_conf["window_size"]
        self.temporal_edges_max_time_diff = model_conf["temporal_edges_max_time_diff"]
        self.temporal_edges_across_views = model_conf["temporal_edges_across_views"]
        self.temporal_edges_max_speed = model_conf["temporal_edges_max_speed"]
        self.temporal_edges_speed_type = model_conf["temporal_edges_speed_type"]
        self.view_edges_max_distance = model_conf["view_edges_max_distance"]
        self.social_edges_max_distance = model_conf["social_edges_max_distance"]

        self.use_camera_node = model_conf["use_camera_node"]
        self.camera_node_features = model_conf["camera_node_features"]
        self.camera_edge_features = model_conf["camera_edge_features"]

        self.use_oracle = model_conf["use_oracle"]
        self.oracle_type = model_conf["oracle_type"]

        self.training = training
        self.drop_node_rate = model_conf["drop_node_rate"]
        self.drop_edge_rate = model_conf["drop_edge_rate"]
        self.mask_node_features_rate = model_conf["mask_node_features_rate"]
        self.mask_edge_features_rate = model_conf["mask_edge_features_rate"]
        self.masking_type = model_conf["masking_type"]

        self.gt_dict = {}  # Initialize gt_dict to store ground truth data

        # populated when computing tracking metrics. used for visualization
        self.pred_ids = None
        self.dset_name = None
        self.sequence = None

        log.spam("HeteroGraph initialized with the following parameters:")
        log.spam(f"  Features: {self.features}")
        log.spam(f"  Window size: {self.window_size}")
        log.spam(f"  Temporal edges:")
        log.spam(f"    Max time difference: {self.temporal_edges_max_time_diff}")
        log.spam(f"    Across views: {self.temporal_edges_across_views}")
        log.spam(f"    Max speed: {self.temporal_edges_max_speed}")
        log.spam(f"    Features: {self.edge_features['temporal']}")
        log.spam(f"  View edges:")
        log.spam(f"    Max distance: {self.view_edges_max_distance}")
        log.spam(f"    Features: {self.edge_features['view']}")
        log.spam(f"  Social edges:")
        log.spam(f"    Features: {self.edge_features['social']}")

    def add_node_attr(self, graph_data, node_type, attr, data):
        if not hasattr(graph_data[node_type], attr):
            setattr(graph_data[node_type], attr, data)
        else:
            setattr(graph_data[node_type], attr, torch.cat([getattr(graph_data[node_type], attr), data]))

    def add_edge_attr(self, graph_data, edge_triplet, attr, data):
        src_node, edge_type, dst_node = edge_triplet

        if not hasattr(graph_data[src_node, edge_type, dst_node], attr):
            setattr(graph_data[src_node, edge_type, dst_node], attr, data)
        else:
            existing_attr = getattr(graph_data[src_node, edge_type, dst_node], attr)

            # Special case for edge_index: concatenate along dim=1
            if attr == 'edge_index':
                new_attr = torch.cat([existing_attr, data], dim=1)
            else:
                new_attr = torch.cat([existing_attr, data], dim=0)

            setattr(graph_data[src_node, edge_type, dst_node], attr, new_attr)

    def update_graph(self, mv_data, model):
        # log.debug(f"Updating graph")
        # Check if adding the new timestep will exceed the window size
        if self.nb_timesteps + 1 > self.window_size:
            self.archive_last_timestep()
        else:
            self.nb_timesteps += 1

        self.save_gt_dict(mv_data)        

        self.add_detections_and_social_edges(mv_data, model)

        if not hasattr(self.data["detection"], "x") or len(self.data["detection"].x) == 0:
            return

        try:
            self.create_view_edges(model, self.view_edges_max_distance)
        except Exception as e:
            import pdb; pdb.set_trace()

        self.create_temporal_edges(model, self.temporal_edges_max_time_diff, self.temporal_edges_across_views, self.temporal_edges_max_speed)

        if self.use_camera_node:
            self.add_camera_nodes(mv_data, model)
            self.create_camera_edges(model)

    #     self.detach_graph()
    #     self.detach_graph(use_archived=True)

    def save_gt_dict(self, mv_data):
        for view_key, view_data in mv_data.items():
            if view_key == 'normalization_dict' or view_key == 'camera_dict':
                continue
            
            timestamp = view_data['gt_dict']['timestamp']
            view_id = view_data['gt_dict']['view_id']
            
            if 'gt_dict' in view_data:
                if timestamp not in self.gt_dict:
                    self.gt_dict[timestamp] = {}
                self.gt_dict[timestamp][view_id] = view_data['gt_dict']

    def detach_graph(self, use_archived=False):
        data_to_detach = self.archived_data if use_archived else self.data

        for node_type in data_to_detach.node_types:
            for attr, value in data_to_detach[node_type].items():
                if isinstance(value, torch.Tensor):
                    data_to_detach[node_type][attr] = value.detach()

        for edge_type in data_to_detach.edge_types:
            for attr, value in data_to_detach[edge_type].items():
                if isinstance(value, torch.Tensor):
                    data_to_detach[edge_type][attr] = value.detach()


    def add_detections_and_social_edges(self, mv_data, model):
        norm_dict = mv_data['normalization_dict']
        for view_key, view_data in mv_data.items():
            if view_key == 'normalization_dict' or view_key == 'camera_dict':
                continue

            # Check if there are any detections in this view
            if view_data['timestamp'].shape[0] == 0:
                log.spam(f"No detections in view {view_key}, skipping")
                continue

            features = self.create_node_features(view_data, norm_dict, model)

            # Apply node dropping if training and drop_node_rate > 0
            if self.training and self.drop_node_rate > 0:
                num_nodes = features.shape[0]
                num_nodes_to_keep = int(num_nodes * (1 - self.drop_node_rate))
                indices_to_keep = torch.randperm(num_nodes)[:num_nodes_to_keep]
                
                features = features[indices_to_keep]
                view_data['timestamp'] = view_data['timestamp'][indices_to_keep]
                view_data['label'] = view_data['label'][indices_to_keep]
                view_data['view_id'] = view_data['view_id'][indices_to_keep]
                view_data['person_id'] = view_data['person_id'][indices_to_keep]
                view_data['bbox'] = view_data['bbox'][indices_to_keep]
                view_data['world_points'] = view_data['world_points'][indices_to_keep]
                view_data['confidence'] = view_data['confidence'][indices_to_keep]

            self.add_node_attr(self.data, 'detection', 'x', features)
            self.add_node_attr(self.data, 'detection', 'timestamp', view_data['timestamp'])
            self.add_node_attr(self.data, 'detection', 'label', view_data['label'])
            self.add_node_attr(self.data, 'detection', 'view_id', view_data['view_id'])
            self.add_node_attr(self.data, 'detection', 'person_id', view_data['person_id'])
            self.add_node_attr(self.data, 'detection', 'bbox', view_data['bbox'])
            self.add_node_attr(self.data, 'detection', 'world_points', view_data['world_points'])
            self.add_node_attr(self.data, 'detection', 'confidence', view_data['confidence'])

            # Create social edges for new detections within this view
            new_detection_indices = range(self.data['detection'].x.shape[0] - features.shape[0], self.data['detection'].x.shape[0])
            self.create_social_edges(new_detection_indices, model, self.social_edges_max_distance)

    def create_social_edges(self, detection_indices, model, max_distance=None):
        num_detections = len(detection_indices)
        
        # Create all possible pairs of indices
        idx = torch.tensor(detection_indices, device=self.device)
        rows, cols = torch.meshgrid(idx, idx, indexing='ij')
        pairs = torch.stack([rows.flatten(), cols.flatten()], dim=1)
        
        # Remove self-loops and duplicate edges
        mask = pairs[:, 0] < pairs[:, 1]
        pairs = pairs[mask]
        
        # Filter out pairs based on max_distance
        if max_distance is not None:
            world_points = self.data['detection'].world_points[detection_indices]
            distances = torch.norm(world_points[pairs[:, 0] - detection_indices[0]] - world_points[pairs[:, 1] - detection_indices[0]], dim=1)
            distance_mask = distances < max_distance
            pairs = pairs[distance_mask]
        
        # Apply edge dropping if training and drop_edge_rate > 0
        if self.training and self.drop_edge_rate > 0:
            num_edges = pairs.shape[0]
            num_edges_to_keep = int(num_edges * (1 - self.drop_edge_rate))
            indices_to_keep = torch.randperm(num_edges)[:num_edges_to_keep]
            pairs = pairs[indices_to_keep]
        
        new_edges = pairs.t()
        
        self.add_edge_attr(
            self.data,
            ('detection', 'social', 'detection'),
            'edge_index',
            new_edges
        )

        edge_features = self.create_edge_features(new_edges, 'social', model)
        
        self.add_edge_attr(
            self.data,
            ('detection', 'social', 'detection'),
            'edge_attr',
            edge_features
        )

    def create_view_edges(self, model, max_distance=None):
        
        num_detections = self.data['detection'].x.shape[0]
        
        latest_timestamp = self.data['detection'].timestamp.max()
        
        latest_mask = self.data['detection'].timestamp == latest_timestamp
        latest_indices = torch.where(latest_mask)[0]
        rows, cols = torch.meshgrid(latest_indices, latest_indices, indexing='ij')
        pairs = torch.stack([rows.flatten(), cols.flatten()], dim=1)
        
        # Remove self-loops and duplicate edges
        mask = pairs[:, 0] < pairs[:, 1]
        pairs = pairs[mask]
        
        view_id = self.data['detection'].view_id
        valid_pairs = view_id[pairs[:, 0]] != view_id[pairs[:, 1]]
        pairs = pairs[valid_pairs]
        
        # Filter out pairs based on max_distance
        if max_distance is not None:
            world_points = self.data['detection'].world_points
            distances = torch.norm(world_points[pairs[:, 0]] - world_points[pairs[:, 1]], dim=1)
            distance_mask = distances < max_distance
            pairs = pairs[distance_mask]
        
        # Apply edge dropping if training and drop_edge_rate > 0
        if self.training and self.drop_edge_rate > 0:
            num_edges = pairs.shape[0]
            num_edges_to_keep = int(num_edges * (1 - self.drop_edge_rate))
            indices_to_keep = torch.randperm(num_edges)[:num_edges_to_keep]
            pairs = pairs[indices_to_keep]

        new_edges = pairs.t()
        
        self.add_edge_attr(
            self.data,
            ('detection', 'view', 'detection'),
            'edge_index',
            new_edges
        )
        
        edge_features = self.create_edge_features(new_edges, 'view', model)
        
        self.add_edge_attr(
            self.data,
            ('detection', 'view', 'detection'),
            'edge_attr',
            edge_features
        )
        
        edge_labels = self.create_edge_labels(new_edges)
        
        self.add_edge_attr(
            self.data,
            ('detection', 'view', 'detection'),
            'edge_label',
            edge_labels
        )

    def create_temporal_edges(self, model, max_time_diff=1, across_views=False, max_speed=None):
        num_detections = self.data['detection'].x.shape[0]
        timestamps = self.data['detection'].timestamp
        current_timestamp = timestamps.max()
        
        # Create pairs of indices, with the second index always from the current timestamp
        current_indices = torch.where(timestamps == current_timestamp)[0]
        cols = current_indices.repeat_interleave(num_detections)
        rows = torch.arange(num_detections, device=self.device).repeat(len(current_indices))
        pairs = torch.stack([rows, cols], dim=1)
        
        # Remove self-loops
        mask = pairs[:, 0] != pairs[:, 1]
        pairs = pairs[mask]
        
        # Filter pairs based on view_id and time difference
        view_id = self.data['detection'].view_id
        time_diff = timestamps[pairs[:, 1]] - timestamps[pairs[:, 0]]
        
        if across_views:
            valid_pairs = (time_diff > 0) & (time_diff <= max_time_diff)
        else:
            valid_pairs = (view_id[pairs[:, 0]] == view_id[pairs[:, 1]]) & \
                        (time_diff > 0) & (time_diff <= max_time_diff)
        
        # Apply max_speed constraint if specified
        if max_speed is not None:
            if self.temporal_edges_speed_type == "world_points":
                world_points = self.data['detection'].world_points
                distances = torch.norm(world_points[pairs[:, 1]] - world_points[pairs[:, 0]], dim=1)
            else:  # bbox
                bbox = self.data['detection'].bbox
                # Get centers of bboxes by taking midpoint of top-left and bottom-right corners
                centers = bbox[:, :2] + (bbox[:, 2:] - bbox[:, :2])/2
                distances = torch.norm(centers[pairs[:, 1]] - centers[pairs[:, 0]], dim=1)
            max_allowed_distance = time_diff * max_speed
            valid_pairs = valid_pairs & (distances <= max_allowed_distance)
        
        pairs = pairs[valid_pairs]

        # Apply edge dropping if training and drop_edge_rate > 0
        if self.training and self.drop_edge_rate > 0:
            num_edges = pairs.shape[0]
            num_edges_to_keep = int(num_edges * (1 - self.drop_edge_rate))
            indices_to_keep = torch.randperm(num_edges)[:num_edges_to_keep]
            pairs = pairs[indices_to_keep]
        
        new_edges = pairs.t()
        
        self.add_edge_attr(
            self.data,
            ('detection', 'temporal', 'detection'),
            'edge_index',
            new_edges
        )
        
        edge_features = self.create_edge_features(new_edges, 'temporal', model)
        
        self.add_edge_attr(
            self.data,
            ('detection', 'temporal', 'detection'),
            'edge_attr',
            edge_features
        )
    
        edge_labels = self.create_edge_labels(new_edges)
        
        self.add_edge_attr(
            self.data,
            ('detection', 'temporal', 'detection'),
            'edge_label',
            edge_labels
        )

    def create_node_features(self, view_data, norm_dict, model):
        features = []
        feature_mapping = OrderedDict()
        current_position = 0

        for feature in self.features:
            if feature in view_data:
                feature_data = view_data[feature]
                if feature in norm_dict:
                    mean = norm_dict[feature]['mean']
                    std = norm_dict[feature]['std']
                    feature_data = (feature_data - mean) / std
                if feature_data.ndim == 1:
                    feature_data = feature_data.unsqueeze(1)

                if feature == 'crops':
                    feature_data = model.initialize_crops(feature_data)

                if self.training and self.mask_node_features_rate > 0:
                    if self.masking_type == 'global':
                        # Mask entire feature with probability mask_node_features_rate
                        if torch.rand(1).item() < self.mask_node_features_rate:
                            feature_data = torch.zeros_like(feature_data)
                    elif self.masking_type == 'partial':
                        # Mask individual elements with probability mask_node_features_rate
                        mask = torch.rand_like(feature_data) < self.mask_node_features_rate
                        feature_data[mask] = 0

                features.append(feature_data)
                
                feature_size = feature_data.shape[1]
                feature_mapping[feature] = (current_position, current_position + feature_size)
                current_position += feature_size

        features = torch.cat(features, dim=-1)
        features = model.initialize_node_features(features, feature_mapping)

        return features

    def create_edge_features(self, edge_index, edge_type, model):
        edge_features = []
        feature_mapping = {}
        current_position = 0

        src_nodes = self.data['detection'].x[edge_index[0]]
        dst_nodes = self.data['detection'].x[edge_index[1]]

        for feature in self.edge_features[edge_type]:
            if feature == 'zeros':
                feature_data = torch.zeros(edge_index.shape[1], 1, device=self.device)
            elif feature == 'noise':
                feature_data = torch.randn(edge_index.shape[1], 16, device=self.device)
            elif feature == 'distance':
                src_world = self.data['detection'].world_points[edge_index[0]]
                dst_world = self.data['detection'].world_points[edge_index[1]]
                feature_data = torch.norm(src_world - dst_world, dim=1, keepdim=True)
            elif feature == 'reid_distance':
                raise NotImplementedError("Not implemented")
                src_features = self.data['detection'].reid_feature[edge_index[0]]
                dst_features = self.data['detection'].reid_feature[edge_index[1]]
                feature_data = torch.linalg.norm(src_features - dst_features, dim=1, ord=2, keepdim=True)
            elif feature == 'elevation':
                src_world = self.data['detection'].world_points[edge_index[0]]
                dst_world = self.data['detection'].world_points[edge_index[1]]
                feature_data = (src_world[:, 2] - dst_world[:, 2]).unsqueeze(1)
            elif feature == 'occlusion':
                raise NotImplementedError("Not implemented")
            elif feature == 'time_distance':
                src_time = self.data['detection'].timestamp[edge_index[0]]
                dst_time = self.data['detection'].timestamp[edge_index[1]]
                feature_data = (dst_time - src_time).unsqueeze(1).float()
            elif feature == 'bboxiou':
                src_bbox = self.data['detection'].bbox[edge_index[0]]
                dst_bbox = self.data['detection'].bbox[edge_index[1]]
                feature_data = calculate_bbox_iou(src_bbox, dst_bbox).unsqueeze(1)
            elif feature == 'bboxratio':
                src_bbox = self.data['detection'].bbox[edge_index[0]]
                dst_bbox = self.data['detection'].bbox[edge_index[1]]
                feature_data = calculate_bbox_ratio(src_bbox, dst_bbox).unsqueeze(1)
            else:
                continue

            if self.training and self.mask_edge_features_rate > 0:
                if self.masking_type == 'global':
                    # Mask entire feature with probability mask_edge_features_rate
                    if torch.rand(1).item() < self.mask_edge_features_rate:
                        feature_data = torch.zeros_like(feature_data)
                elif self.masking_type == 'partial':
                    # Mask individual elements with probability mask_edge_features_rate
                    mask = torch.rand_like(feature_data) < self.mask_edge_features_rate
                    feature_data[mask] = 0

            edge_features.append(feature_data)
            feature_size = feature_data.shape[1]
            feature_mapping[feature] = (current_position, current_position + feature_size)
            current_position += feature_size

        edge_features = torch.cat(edge_features, dim=1) if edge_features else torch.empty(0)

        edge_features = model.initialize_edge_features(edge_features, feature_mapping, edge_type)

        return edge_features

    def create_edge_labels(self, edge_index):
        src_ids = self.data['detection'].person_id[edge_index[0]]
        dst_ids = self.data['detection'].person_id[edge_index[1]]
        labels = ((src_ids == dst_ids) & (src_ids != -1) & (dst_ids != -1)).long()
        return labels

    
    def add_camera_nodes(self, mv_data, model):
        camera_dict = mv_data['camera_dict']
        for view_id, camera_info in camera_dict.items():

            # Check if there are any detections in this view
            if mv_data[view_id]['timestamp'].shape[0] == 0:
                log.spam(f"No detections in view {view_id}, skipping")
                continue

            camera_hash = camera_info['hash']
            timestamp = torch.tensor([mv_data[view_id]['timestamp'][0]], device=self.device)

            # Check if the camera is already in the graph
            if hasattr(self.data['camera'], 'hash'):
                existing_camera_mask = (self.data['camera'].hash == camera_hash)
                if existing_camera_mask.any():
                    # Update timestamp for existing camera
                    existing_camera_index = existing_camera_mask.nonzero().item()
                    self.data['camera'].timestamp[existing_camera_index] = timestamp
                    continue

            # Create new camera node
            camera_features = self.create_camera_features(camera_info, model)
            
            self.add_node_attr(self.data, 'camera', 'x', camera_features)
            self.add_node_attr(self.data, 'camera', 'timestamp', timestamp)
            self.add_node_attr(self.data, 'camera', 'view_id', torch.tensor([view_id], device=self.device))
            self.add_node_attr(self.data, 'camera', 'hash', camera_hash)
            self.add_node_attr(self.data, 'camera', 'position', camera_info['position'].reshape(1, 3))
            self.add_node_attr(self.data, 'camera', 'axis', camera_info['axis'].reshape(1, 3))

    def create_camera_features(self, camera_info, model):
        features = []
        feature_mapping = {}
        current_position = 0

        for feature_name, feature_config in self.camera_node_features.items():
            if feature_name == 'position':
                feature_data = camera_info['position']
            elif feature_name == 'rotation':
                feature_data = camera_info['rotation'].flatten()
            elif feature_name == 'translation':
                feature_data = camera_info['translation'].flatten()
            elif feature_name == 'intrinsic':
                feature_data = camera_info['intrinsic'].flatten()
            elif feature_name == 'frame_size':
                feature_data = camera_info['frame_size']
            elif feature_name == 'axis':
                feature_data = camera_info['axis']
            elif feature_name == 'zeros':
                feature_data = torch.zeros(feature_config['hidden_size'], device=self.device)
            else:
                log.warning(f"Unknown camera feature: {feature_name}")
                continue

            features.append(feature_data)
            feature_size = feature_data.shape[0]
            feature_mapping[feature_name] = (current_position, current_position + feature_size)
            current_position += feature_size

        # Concatenate all features
        features = torch.cat(features).unsqueeze(0)

        # Use model to initialize camera features
        features = model.initialize_camera_features(features, feature_mapping, self.camera_node_features)

        return features


    def create_camera_edges(self, model):
        self.create_camera_next_edges(model)
        self.create_camera_see_edges(model)

    def create_camera_next_edges(self, model):
        camera_timestamps = self.data['camera'].timestamp
        camera_view_ids = self.data['camera'].view_id

        # Find the most recent timestamp
        most_recent_timestamp = camera_timestamps.max()

        # Get indices of cameras with the most recent timestamp
        recent_camera_indices = (camera_timestamps == most_recent_timestamp).nonzero().squeeze(1)

        edge_indices = []
        for i in recent_camera_indices:
            for j in range(len(camera_timestamps)):
                if i != j:
                    # Check if edge_index exists
                    if 'edge_index' in self.data['camera', 'next', 'camera']:
                        # Check if edge already exists
                        existing_edge = ((self.data['camera', 'next', 'camera'].edge_index[0] == i) & 
                                         (self.data['camera', 'next', 'camera'].edge_index[1] == j))
                        if not existing_edge.any():
                            # Add new edge index
                            edge_indices.append([i, j])
                    else:
                        # Add new edge index if edge_index does not exist
                        edge_indices.append([i, j])

        if edge_indices:
            new_edge_index = torch.tensor(edge_indices, device=self.device).t().long()
            self.add_edge_attr(self.data, ('camera', 'next', 'camera'), 'edge_index', new_edge_index)

            # Generate and initialize edge features for all new edges at once
            edge_attr = self.generate_camera_edge_features('next', model, new_edge_index)
            self.add_edge_attr(self.data, ('camera', 'next', 'camera'), 'edge_attr', edge_attr)

    def create_camera_see_edges(self, model):
        camera_timestamps = self.data['camera'].timestamp
        camera_view_ids = self.data['camera'].view_id

        detection_timestamps = self.data['detection'].timestamp
        detection_view_ids = self.data['detection'].view_id

        # Find the most recent timestamp
        most_recent_timestamp = camera_timestamps.max()

        # Get indices of cameras with the most recent timestamp
        recent_camera_indices = (camera_timestamps == most_recent_timestamp).nonzero().squeeze(1)

        camera_edges = []

        for i in recent_camera_indices:
            cam_timestamp = camera_timestamps[i]
            cam_view_id = camera_view_ids[i]
            matching_detections = (detection_timestamps == cam_timestamp) & (detection_view_ids == cam_view_id)
            matching_indices = matching_detections.nonzero().squeeze(1)

            for j in matching_indices:
                camera_edges.append([i, j])

        if camera_edges:
            camera_edges = torch.tensor(camera_edges, device=self.device).t()

            # Apply edge dropping if training and drop_edge_rate > 0
            if self.training and self.drop_edge_rate > 0:
                num_edges = camera_edges.shape[1]
                num_edges_to_keep = int(num_edges * (1 - self.drop_edge_rate))
                perm = torch.randperm(num_edges, device=self.device)
                camera_edges = camera_edges[:, perm[:num_edges_to_keep]]

            self.add_edge_attr(self.data, ('camera', 'see', 'detection'), 'edge_index', camera_edges)
            
            # Generate and initialize edge features for all edges at once
            camera_edge_attrs = self.generate_camera_edge_features('see', model, camera_edges)
            self.add_edge_attr(self.data, ('camera', 'see', 'detection'), 'edge_attr', camera_edge_attrs)

    def generate_camera_edge_features(self, edge_type, model, edge_index):
        if edge_type == 'next':
            # Initialize with zeros for 'next' edges
            num_edges = edge_index.shape[1]
            edge_attr = torch.zeros(num_edges, 1, device=self.device)
            feature_mapping = {f'zeros': (0, 1)}
        elif edge_type == 'see':
            edge_features = []
            feature_mapping = {}
            current_position = 0

            for feature in self.camera_edge_features:
                if feature == 'zeros':
                    feature_data = torch.zeros(edge_index.shape[1], 1, device=self.device)
                elif feature == 'distance':
                    camera_positions = self.data['camera'].position[edge_index[0]]
                    detection_positions = self.data['detection'].world_points[edge_index[1]]
                    feature_data = torch.norm(camera_positions - detection_positions, dim=1, keepdim=True)
                elif feature == 'angle':
                    camera_axes = self.data['camera'].axis[edge_index[0]]
                    camera_positions = self.data['camera'].position[edge_index[0]]
                    detection_positions = self.data['detection'].world_points[edge_index[1]]
                    direction_vectors = detection_positions - camera_positions
                    direction_vectors = torch.nn.functional.normalize(direction_vectors, dim=1)
                    feature_data = torch.sum(camera_axes * direction_vectors, dim=1, keepdim=True)
                elif feature == 'occlusions':
                    from misc.occlusions import camera_person_occlusion
                    camera_positions = self.data['camera'].position[edge_index[0]]
                    person_feet_positions = self.data['detection'].world_points[edge_index[1]]
                    
                    # Create a list to store occlusion features for each edge
                    occlusion_features = []
                    for i in range(edge_index.shape[1]):
                        # Get occlusion scores for feet, mid-body, and head
                        occlusion_scores = camera_person_occlusion(camera_positions[i].detach().cpu().numpy(), person_feet_positions[i].detach().cpu().numpy())
                        occlusion_features.append(occlusion_scores)
                    
                    # Stack all occlusion features into a tensor
                    if len(occlusion_features) > 0:
                        feature_data = torch.stack(occlusion_features, dim=0).to(self.device)
                    else:
                        feature_data = torch.zeros(edge_index.shape[1], 3, device=self.device)
                elif feature == 'self_occlusion':
                    # This is a placeholder. Actual occlusion calculation would be more complex.
                    feature_data = torch.rand(edge_index.shape[1], 1, device=self.device)
                else:
                    continue

                if self.training and self.mask_edge_features_rate > 0:
                    if self.masking_type == 'global':
                        if torch.rand(1).item() < self.mask_edge_features_rate:
                            feature_data = torch.zeros_like(feature_data)
                    elif self.masking_type == 'partial':
                        mask = torch.rand_like(feature_data) < self.mask_edge_features_rate
                        feature_data[mask] = 0

                edge_features.append(feature_data)
                feature_size = feature_data.shape[1]
                feature_mapping[feature] = (current_position, current_position + feature_size)
                current_position += feature_size

            edge_attr = torch.cat(edge_features, dim=1) if edge_features else torch.empty(0)
        else:
            raise ValueError(f"Unknown edge type: {edge_type}")

        edge_attr = model.initialize_camera_edge_features(edge_attr, feature_mapping, edge_type)

        return edge_attr


    def archive_last_timestep(self):
        # Move the oldest timestep data to archived_data

        if not hasattr(self.data["detection"], "timestamp") or len(self.data["detection"].timestamp) == 0:
            return

        oldest_timestamp = self.data['detection'].timestamp.min()
        previous_archived_detections_count = len(self.archived_data['detection'].bbox) if hasattr(self.archived_data['detection'], 'bbox') else 0
        mask = self.data['detection'].timestamp > oldest_timestamp
        
        # Move detection metadata, labels, and predictions to archived_data
        self.add_node_attr(self.archived_data, 'detection', 'timestamp', self.data['detection'].timestamp[~mask].detach().cpu())
        self.add_node_attr(self.archived_data, 'detection', 'view_id', self.data['detection'].view_id[~mask].detach().cpu())
        self.add_node_attr(self.archived_data, 'detection', 'person_id', self.data['detection'].person_id[~mask].detach().cpu())
        self.add_node_attr(self.archived_data, 'detection', 'bbox', self.data['detection'].bbox[~mask].detach().cpu())
        self.add_node_attr(self.archived_data, 'detection', 'world_points', self.data['detection'].world_points[~mask].detach().cpu())
        self.add_node_attr(self.archived_data, 'detection', 'label', self.data['detection'].label[~mask].detach().cpu())
        self.add_node_attr(self.archived_data, 'detection', 'confidence', self.data['detection'].confidence[~mask].detach().cpu())
        if hasattr(self.data['detection'], 'pred'):
            self.add_node_attr(self.archived_data, 'detection', 'pred', self.data['detection'].pred[~mask].detach().cpu())
        
        # Remove from current data
        self.data['detection'].x = self.data['detection'].x[mask]
        self.data['detection'].timestamp = self.data['detection'].timestamp[mask]
        self.data['detection'].view_id = self.data['detection'].view_id[mask]
        self.data['detection'].person_id = self.data['detection'].person_id[mask]
        self.data['detection'].bbox = self.data['detection'].bbox[mask]
        self.data['detection'].world_points = self.data['detection'].world_points[mask]
        self.data['detection'].label = self.data['detection'].label[mask]
        self.data['detection'].confidence = self.data['detection'].confidence[mask]
        if hasattr(self.data['detection'], 'pred'):
            self.data['detection'].pred = self.data['detection'].pred[mask]
        
        camera_mask = None
        previous_archived_cameras_count = 0
        
        if self.use_camera_node:
            # Archive camera data
            camera_mask = self.data['camera'].timestamp > oldest_timestamp
            previous_archived_cameras_count = len(self.archived_data['camera'].hash) if hasattr(self.archived_data['camera'], 'hash') else 0
            
            # Move camera metadata to archived_data
            self.add_node_attr(self.archived_data, 'camera', 'timestamp', self.data['camera'].timestamp[~camera_mask].detach().cpu())
            self.add_node_attr(self.archived_data, 'camera', 'view_id', self.data['camera'].view_id[~camera_mask].detach().cpu())
            self.add_node_attr(self.archived_data, 'camera', 'hash', self.data['camera'].hash[~camera_mask].detach().cpu())
            self.add_node_attr(self.archived_data, 'camera', 'position', self.data['camera'].position[~camera_mask].detach().cpu())
            self.add_node_attr(self.archived_data, 'camera', 'axis', self.data['camera'].axis[~camera_mask].detach().cpu())
            if hasattr(self.data['camera'], 'x'):
                self.add_node_attr(self.archived_data, 'camera', 'x', self.data['camera'].x[~camera_mask].detach().cpu())
            
            # Remove from current data
            self.data['camera'].timestamp = self.data['camera'].timestamp[camera_mask]
            self.data['camera'].view_id = self.data['camera'].view_id[camera_mask]
            self.data['camera'].hash = self.data['camera'].hash[camera_mask]
            self.data['camera'].position = self.data['camera'].position[camera_mask]
            self.data['camera'].axis = self.data['camera'].axis[camera_mask]
            if hasattr(self.data['camera'], 'x'):
                self.data['camera'].x = self.data['camera'].x[camera_mask]
        
        # Update edges
        self.archive_edges(mask, previous_archived_detections_count, camera_mask, previous_archived_cameras_count)
        

    def archive_edges(self, mask, previous_archived_detections_count, camera_mask, previous_archived_cameras_count):
        # Create a mapping from old indices to new indices
        old_to_new_detection = torch.cumsum(mask, dim=0) - 1
        old_to_new_detection[~mask] = -1  # Set removed vertices to -1

        # Archive detection-to-detection edges
        for edge_type in ['social', 'view', 'temporal']:
            self.archive_detection_edges(edge_type, mask, old_to_new_detection, previous_archived_detections_count)

        if self.use_camera_node:
            old_to_new_camera = torch.cumsum(camera_mask, dim=0) - 1
            old_to_new_camera[~camera_mask] = -1  # Set removed cameras to -1

            # Archive camera-to-detection edges (see)
            self.archive_camera_detection_edges('see', mask, camera_mask, old_to_new_detection, old_to_new_camera, previous_archived_detections_count, previous_archived_cameras_count)

            # Archive camera-to-camera edges (next)
            self.archive_camera_camera_edges('next', camera_mask, old_to_new_camera, previous_archived_cameras_count)

    def archive_detection_edges(self, edge_type, mask, old_to_new, previous_archived_detections_count):
        edge_index = self.data['detection', edge_type, 'detection'].edge_index
        valid_edges = mask[edge_index[0]] & mask[edge_index[1]]
        
        # Update the edge indices for current data
        new_edge_index = old_to_new[edge_index[:, valid_edges]]
        nb_vert = mask.sum()
        if ((new_edge_index < 0) | (new_edge_index >= nb_vert)).any():
            raise ValueError("Some edges are pointing to non-existing vertices after archiving.")

        # Archive edges to be removed
        archived_edge_index = edge_index[:, ~valid_edges].detach().cpu()
        # Shift the detection ids in archived_edge_index
        archived_edge_index += previous_archived_detections_count

        if hasattr(self.archived_data['detection', edge_type, 'detection'], 'edge_index'):
            self.archived_data['detection', edge_type, 'detection'].edge_index = torch.cat([
                self.archived_data['detection', edge_type, 'detection'].edge_index,
                archived_edge_index
            ], dim=1)
        else:
            self.archived_data['detection', edge_type, 'detection'].edge_index = archived_edge_index

        # Update current data
        self.data['detection', edge_type, 'detection'].edge_index = new_edge_index

        # Archive and update edge attributes
        for attr in ['edge_attr', 'edge_label', 'edge_pred']:
            if hasattr(self.data['detection', edge_type, 'detection'], attr):
                current_attr = getattr(self.data['detection', edge_type, 'detection'], attr)
                
                # Archive attributes of removed edges
                archived_attr = current_attr[~valid_edges].detach().cpu()
                if hasattr(self.archived_data['detection', edge_type, 'detection'], attr):
                    archived_attr = torch.cat([
                        getattr(self.archived_data['detection', edge_type, 'detection'], attr),
                        archived_attr
                    ], dim=0)
                
                # Edge features are not stored in archived data
                if attr != 'edge_attr':
                    setattr(self.archived_data['detection', edge_type, 'detection'], attr, archived_attr)
                
                # Remove edge attr from current data
                setattr(self.data['detection', edge_type, 'detection'], attr, current_attr[valid_edges])

    def archive_camera_detection_edges(self, edge_type, detection_mask, camera_mask, old_to_new_detection, old_to_new_camera, previous_archived_detections_count, previous_archived_cameras_count):
        edge_index = self.data['camera', edge_type, 'detection'].edge_index
        valid_edges = camera_mask[edge_index[0]] & detection_mask[edge_index[1]]
        
        # Update the edge indices for current data
        new_edge_index = edge_index[:, valid_edges].clone()
        new_edge_index[0] = old_to_new_camera[new_edge_index[0]]
        new_edge_index[1] = old_to_new_detection[new_edge_index[1]]
        
        # Archive edges to be removed
        archived_edge_index = edge_index[:, ~valid_edges].detach().cpu()
        archived_edge_index[0] += previous_archived_cameras_count
        archived_edge_index[1] += previous_archived_detections_count

        if hasattr(self.archived_data['camera', edge_type, 'detection'], 'edge_index'):
            self.archived_data['camera', edge_type, 'detection'].edge_index = torch.cat([
                self.archived_data['camera', edge_type, 'detection'].edge_index,
                archived_edge_index
            ], dim=1)
        else:
            self.archived_data['camera', edge_type, 'detection'].edge_index = archived_edge_index

        # Update current data
        self.data['camera', edge_type, 'detection'].edge_index = new_edge_index

        # Archive and update edge attributes
        for attr in ['edge_attr']:
            if hasattr(self.data['camera', edge_type, 'detection'], attr):
                current_attr = getattr(self.data['camera', edge_type, 'detection'], attr)
                
                # Archive attributes of removed edges
                archived_attr = current_attr[~valid_edges].detach().cpu()
                if hasattr(self.archived_data['camera', edge_type, 'detection'], attr):
                    archived_attr = torch.cat([
                        getattr(self.archived_data['camera', edge_type, 'detection'], attr),
                        archived_attr
                    ], dim=0)
                
                # Edge features are not stored in archived data
                if attr != 'edge_attr':
                    setattr(self.archived_data['camera', edge_type, 'detection'], attr, archived_attr)
                
                # Remove edge attr from current data
                setattr(self.data['camera', edge_type, 'detection'], attr, current_attr[valid_edges])

    def archive_camera_camera_edges(self, edge_type, camera_mask, old_to_new_camera, previous_archived_cameras_count):
        if not hasattr(self.data['camera', edge_type, 'camera'], 'edge_index'):
            return
        edge_index = self.data['camera', edge_type, 'camera'].edge_index
        valid_edges = camera_mask[edge_index[0]] & camera_mask[edge_index[1]]
        
        # Update the edge indices for current data
        new_edge_index = old_to_new_camera[edge_index[:, valid_edges]]
        
        # Archive edges to be removed
        archived_edge_index = edge_index[:, ~valid_edges].detach().cpu()
        archived_edge_index += previous_archived_cameras_count

        if hasattr(self.archived_data['camera', edge_type, 'camera'], 'edge_index'):
            self.archived_data['camera', edge_type, 'camera'].edge_index = torch.cat([
                self.archived_data['camera', edge_type, 'camera'].edge_index,
                archived_edge_index
            ], dim=1)
        else:
            self.archived_data['camera', edge_type, 'camera'].edge_index = archived_edge_index

        # Update current data
        self.data['camera', edge_type, 'camera'].edge_index = new_edge_index

        # Archive and update edge attributes
        for attr in ['edge_attr']:
            if hasattr(self.data['camera', edge_type, 'camera'], attr):
                current_attr = getattr(self.data['camera', edge_type, 'camera'], attr)
                
                # Archive attributes of removed edges
                archived_attr = current_attr[~valid_edges].detach().cpu()
                if hasattr(self.archived_data['camera', edge_type, 'camera'], attr):
                    archived_attr = torch.cat([
                        getattr(self.archived_data['camera', edge_type, 'camera'], attr),
                        archived_attr
                    ], dim=0)
                
                # Edge features are not stored in archived data
                if attr != 'edge_attr':
                    setattr(self.archived_data['camera', edge_type, 'camera'], attr, archived_attr)
                
                # Remove edge attr from current data
                setattr(self.data['camera', edge_type, 'camera'], attr, current_attr[valid_edges])
    def archive_remaining_data(self):
        while self.data['detection'].timestamp.shape[0] > 0:
            self.archive_last_timestep()
            # log.debug(f"Archiving last timestep, {self.data['detection'].timestamp.shape[0]} detections remaining")

        
        # Ensure all data has been archived
        assert self.data['detection'].timestamp.shape[0] == 0, "Failed to archive all data"
        self.nb_timesteps = 0
        
    def compute_edge_temporal_lengths(self, use_archived=False):
        """
        Compute and log the distribution of temporal lengths (timestamp differences) for edges of different types.
        
        Args:
            use_archived (bool): Whether to use archived data or current data
        """
        data = self.archived_data if use_archived else self.data

        # Process each edge type that connects detections
        for edge_type in ['temporal', 'view', 'social']:
            edge_key = ('detection', edge_type, 'detection')
            if edge_key in data.edge_types:
                # Get edge indices
                edge_index = data[edge_key].edge_index
                if edge_index.shape[1] == 0:  # Skip if no edges
                    continue
                    
                # Get timestamps for source and destination nodes
                src_timestamps = data['detection'].timestamp[edge_index[0]]
                dst_timestamps = data['detection'].timestamp[edge_index[1]]
                
                # Compute absolute time differences
                time_diffs = torch.abs(dst_timestamps - src_timestamps)
                
                # Count unique temporal lengths
                unique_lengths, counts = torch.unique(time_diffs, return_counts=True)
                
                # Log the distribution
                log.info(f"Temporal length distribution for {edge_type} edges:")
                for length, count in zip(unique_lengths.tolist(), counts.tolist()):
                    log.info(f"  Length {length}: {count} edges")
    
    def get_metrics(self, use_archived=False):
        metrics = {}
        data = self.archived_data if use_archived else self.data

        # Node classification metrics
        if hasattr(data['detection'], 'label') and hasattr(data['detection'], 'pred'):
            node_metrics = calculate_classification_metrics(
                data['detection'].label, 
                torch.sigmoid(data['detection'].pred), 
                prefix='node_'
            )
            metrics.update(node_metrics)

        # Edge classification metrics
        for edge_type in ['temporal', 'view']:
            edge_key = ('detection', edge_type, 'detection')
            if edge_key in data.edge_types:
                edge_data = data[edge_key]
                if hasattr(edge_data, 'edge_label') and hasattr(edge_data, 'edge_pred'):
                    edge_metrics = calculate_classification_metrics(
                        edge_data.edge_label,
                        torch.sigmoid(edge_data.edge_pred),
                        prefix=f'{edge_type}_edge_'
                    )
                    metrics.update(edge_metrics)

        return metrics

    def save_gt_dict_to_json(self, json_path):
        """
        Save the gt_dict to a JSON file.
        
        Args:
            json_path (str): Path to save the JSON file.
        """
        import json
        
        # Convert numpy arrays and tensors to lists
        serializable_gt_dict = {}
        for timestamp, view_data in self.gt_dict.items():
            serializable_gt_dict[str(timestamp)] = {}
            for view_id, gt_info in view_data.items():
                serializable_gt_dict[str(timestamp)][str(view_id)] = {
                    "bbox": gt_info["bbox"].tolist(),
                    "person_id": gt_info["person_id"].tolist(),
                    "world_points": gt_info["world_points"].tolist(),
                    "timestamp": gt_info["timestamp"],
                    "view_id": gt_info["view_id"]
                }
        
        with open(json_path, 'w') as f:
            json.dump(serializable_gt_dict, f)
        
        log.info(f"Saved gt_dict to {json_path}")

    def load_gt_dict_from_json(self, json_path):
        """
        Load the gt_dict from a JSON file.
        
        Args:
            json_path (str): Path to load the JSON file from.
        """
        import json
        import torch
        
        with open(json_path, 'r') as f:
            loaded_gt_dict = json.load(f)
        
        # Convert lists back to tensors
        self.gt_dict = {}
        for timestamp, view_data in loaded_gt_dict.items():
            self.gt_dict[int(timestamp)] = {}
            for view_id, gt_info in view_data.items():
                self.gt_dict[int(timestamp)][int(view_id)] = {
                    "bbox": torch.tensor(gt_info["bbox"]),
                    "person_id": torch.tensor(gt_info["person_id"]),
                    "world_points": torch.tensor(gt_info["world_points"]),
                    "timestamp": gt_info["timestamp"],
                    "view_id": gt_info["view_id"]
                }
        
        log.info(f"Loaded gt_dict from {json_path}")


    def get_tracking_metrics(self, use_archived=True, use_engin_greedy=False, dset_name=None, sequence=None, mot_metric_evaluator=None, metric_threshold=100):
        data = self.archived_data if use_archived else self.data
        
        if use_engin_greedy:
            from graph import postprocessing_engin
            pred_ids = postprocessing_engin.extract_multiview_trajectories_engin(data)
        else:
            # Extract multiview trajectories
            _, pred_ids = self.extract_multiview_trajectories(use_archived=use_archived, use_oracle=self.use_oracle)
        # Log statistics about extracted trajectories
        num_trajectories = len(set(pred_ids.tolist()))
        num_detections = len(pred_ids)
        num_unassigned = (pred_ids == -1).sum().item()
        
        log.info(f"Trajectory Statistics:")
        log.info(f"  Total trajectories: {num_trajectories}")
        log.info(f"  Total detections: {num_detections}")
        log.info(f"  Unassigned detections: {num_unassigned}")
        log.info(f"  Average trajectory length: {(num_detections - num_unassigned) / max(num_trajectories, 1):.1f}")


        self.pred_ids = pred_ids
        self.dset_name = dset_name
        self.sequence = sequence
        
        if mot_metric_evaluator is not None:
            gt_data, pred_data = mot_metric_evaluator.save_sequence_data(data, pred_ids, self.gt_dict, dset_name, sequence)
            return None, gt_data, pred_data
        else:
            return get_tracking_metrics(data, pred_ids, self.gt_dict, metric_threshold)

        
    def compute_graph_cost(self, use_archived=True, type="pred", min_prob=1e-15):
        data = self.archived_data if use_archived else self.data

        if type == "pred":
            graph_cost = compute_graph_cost_from_id(data, self.pred_ids)
            log.debug(f"Pred shape: {self.pred_ids.shape}")
        elif type == "gt":
            graph_cost = compute_graph_cost_from_id(data, data['detection'].label)
            log.debug(f"Gt shape: {data['detection'].label.shape}")
        elif type == "lower_bound":
            # Apply sigmoid to get probabilities
            temporal_probs = torch.clamp(torch.sigmoid(data[('detection', 'temporal', 'detection')].edge_pred), min=min_prob, max=1-min_prob)
            view_probs = torch.clamp(torch.sigmoid(data[('detection', 'view', 'detection')].edge_pred), min=min_prob, max=1-min_prob)
            
            # Compute negative log likelihood
            temporal_nll = torch.clamp(-torch.log(temporal_probs / torch.clamp(1 - temporal_probs, min=min_prob)), max=0)
            view_nll = torch.clamp(-torch.log(view_probs / torch.clamp(1 - view_probs, min=min_prob)), max=0)

            graph_cost = temporal_nll.sum() + view_nll.sum()            

        else:
            raise ValueError(f"Invalid type: {type}")

        return graph_cost

    def visualize_graph(self, show_views=None, display_temp_edges=True, display_view_edges=True, display_social_edges=True, start_time=None, end_time=None, pdf_filename=None, use_archived=False, only_positive_edges=False):
        data = self.archived_data if use_archived else self.data

        return visualization.extract_data_and_visualize_graph(data, show_views, display_temp_edges, display_view_edges, display_social_edges, start_time, end_time, pdf_filename, only_positive_edges)

    def visualize_single_view(self, selected_view, display_temp_edges=False, display_social_edges=False, pdf_filename=None, last_timestamp=False, frame_image_path=None, use_archived=False):
        data = self.archived_data if use_archived else self.data
        
        return visualization.extract_data_and_visualize_single_view(data, selected_view, display_temp_edges, display_social_edges, pdf_filename, last_timestamp, frame_image_path)

    def visualize_sequence_predictions(self, mp4_filename, selected_views, use_archived=True, show_pred=True, show_gt=True):
        data = self.archived_data if use_archived else self.data
        
        return visualization.visualize_sequence_predictions(data, self.pred_ids, self.gt_dict, self.dset_name, self.sequence, mp4_filename, show_pred, show_gt, selected_views)

    def extract_multiview_trajectories(self, use_archived=False, use_oracle=False):
        data = self.archived_data if use_archived else self.data
        
        import time
        start_time = time.time()
        
        if use_oracle:
            log.info(f"Extracting oracle trajectories with type {self.oracle_type}")
            if self.oracle_type == "pred_id":
                result = postprocessing.extract_oracles_trajectories_pred_id(data)
            elif self.oracle_type == "graph_edges":
                result = postprocessing.extract_oracles_trajectories(data)
            elif self.oracle_type == "bytetrack" or self.oracle_type == "sort" or self.oracle_type == "bytetrackMV":
                result = postprocessing.extract_tracjectory_with_tracker(data, self.oracle_type)
            else:
                raise ValueError(f"Invalid oracle type: {self.oracle_type}")
        else:
            result = postprocessing.extract_multiview_trajectories(data)
            
        end_time = time.time()
        log.info(f"Postprocessing took {end_time - start_time:.2f} seconds")
        
        return result

    def export_graph_as_json(self, json_path, use_archived=False):
        data = self.archived_data if use_archived else self.data

        postprocessing.export_graph_as_json(data, json_path)

    def get_stats(self, use_archived=False, print_log=False):
        data = self.archived_data if use_archived else self.data

        return get_graph_stats(data, print_log=print_log)

    def to(self, device):
        self.data = self.data.to(device)
        
        return self


    def __len__(self):
        return len(self.vertices)

