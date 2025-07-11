import os
import sys
import argparse
import time
from configs.pathes import conf_path, data_path, model_path
from configs.utils import read_yaml_file, convert_yaml_dict_to_arg_list, fill_dict_with_missing_value, aug_tuple, feature_tuple, args_to_dict, FeatureDictAction
from misc.log_utils import log, set_log_level, add_log_file_handler    


parser = argparse.ArgumentParser(description='Unified People Tracking with Graph Neural Networks - Multi-object tracking across multiple camera views using heterogeneous graphs.')

parser_train = parser.add_argument_group("training")
parser_data = parser.add_argument_group("dataset")
parser_model = parser.add_argument_group("model")
parser_loss = parser.add_argument_group("loss")


####### Configuration #######
parser.add_argument("-n", '--name', default="model", 
                   help='Experiment name for saving logs and checkpoints. Used to identify this training run.')

parser.add_argument("-pf", "--print_frequency", dest="print_frequency", type=int, default=100,
                   help="Print training progress every N iterations. Lower values give more frequent updates but slower training.")

parser.add_argument("-dev", "--device", dest="device", default="cuda", choices=["cuda", "cpu"],
                   help="Device for training/inference. Use 'cuda' for GPU acceleration, 'cpu' for CPU-only execution.")

parser.add_argument("-ch", "--nb_checkpoint", dest="nb_checkpoint", default=10, type=int,
                   help="Maximum number of checkpoints to keep. Older checkpoints are automatically deleted to save disk space.")

parser.add_argument("-l", "--log_lvl", dest="log_lvl", default="debug", 
                   choices=["debug", "spam", "verbose", "info", "warning", "error"],
                   help="Logging verbosity level. Use 'debug' for development, 'info' for production training.")

parser.add_argument("-cfg", "--config_file", dest="config_file", default=None, type=str,
                   help="Path to YAML configuration file with default values. Command line arguments override config file values.")

parser.add_argument("-ckp", "--checkpoint", dest="checkpoint", default=None,
                   help="Path to checkpoint file for resuming training or starting evaluation. Automatically resumes from last epoch.")

parser.add_argument("-ovc", "--override_conf", dest="override_conf", action="store_true", default=False,
                   help="Use configuration from checkpoint file instead of current config. Useful for exact reproduction of previous experiments.")


####### Training #######
parser_train.add_argument("-lr", "--learning_rate", dest="lr", type=float, default=0.0001,
                         help="Initial learning rate for Adam optimizer. Typical range: 1e-5 to 1e-3. Lower values for fine-tuning.")

parser_train.add_argument("-lrd", "--learning_rate_decrease", dest="lrd", nargs='+', type=float, 
                         default=[0.5, 20, 40, 60, 80, 100],
                         help="Learning rate schedule: [multiplier, epoch1, epoch2, ...]. LR is multiplied by first value at specified epochs.")

parser_train.add_argument("-dec", "--decay", dest="decay", type=float, default=5e-4,
                         help="L2 weight decay for Adam optimizer. Helps prevent overfitting. Typical range: 1e-5 to 1e-3.")

parser_train.add_argument("-agi", '--accumulate_grad_iter', dest="accumulate_grad_iter", type=int, default=1,
                         help="Gradient accumulation steps. Effectively increases batch size by this factor. Use for large models or limited GPU memory.")

parser_train.add_argument("-metp", "--metric_to_print", dest="metric_to_print", nargs='+', type=str,
                         default=["node_f1_score", "temporal_edge_f1_score", "view_edge_f1_score"],
                         help="Metrics to log during validation. Available: node_f1_score, temporal_edge_f1_score, view_edge_f1_score.")

parser_train.add_argument("-lott", "--loss_to_print", dest="loss_to_print", nargs='+', type=str,
                         default=["loss_node", "loss_view", "loss_temporal", "loss_temporal_sigmoid"],
                         help="Loss components to log during training. Helps monitor which parts of the model are learning.")

parser_train.add_argument("-max_epoch", "--max_epoch", dest="max_epoch", type=int, default=200,
                         help="Maximum training epochs. Training stops early if validation metrics plateau.")

parser_train.add_argument("-cm2d", "--compute_metric_in_2d", dest="compute_metric_in_2d", action="store_true", default=False,
                         help="Compute tracking metrics in 2D image space instead of 3D world coordinates. Use for datasets without 3D ground truth.")

parser_train.add_argument("-gvis", "--generate_visualization", dest="generate_visualization", action="store_true", default=False,
                         help="Generate graph visualizations during evaluation. Useful for debugging and understanding model predictions.")

parser_train.add_argument("-gvid", "--generate_video", dest="generate_video", action="store_true", default=False,
                         help="Generate tracking result videos during evaluation. Enables visual verification of tracking quality.")

parser_train.add_argument("-gexp", "--export_graph", dest="export_graph", action="store_true", default=False,
                         help="Export graph structure as JSON files. Useful for analysis and debugging of graph construction.")

parser_train.add_argument("-egt", "--export_gt", dest="export_gt", action="store_true", default=False,
                         help="Export ground truth annotations as JSON files. Useful for data analysis and external evaluation tools.")


####### Dataset Configuration #######
parser_data.add_argument("-nw", "--num_workers", dest="num_workers", type=int, default=2,
                        help="Number of parallel data loading workers. Increase for faster data loading, but uses more memory.")

parser_data.add_argument("-shft", "--shuffle_train", dest="shuffle_train", action="store_true", default=False,
                        help="Shuffle training data every epoch. Generally recommended for better convergence.")

parser_data.add_argument("-bs", "--batch_size", dest="batch_size", type=int, default=1,
                        help="Batch size for training and evaluation. Currently only supports batch_size=1 due to variable graph sizes.")

parser_data.add_argument("-tdset", "--train_datasets", dest="train_datasets", type=str, nargs='+', default=[],
                        choices=["wildtrack", "mot17train", "mot20train", "scouttrain", "scoutval", "scoutmonotrain", "scoutmonoval"],
                        help="Training datasets. Multiple datasets can be combined. Use 'wildtrack' for multi-view, 'mot17train'/'mot20train' for single-view.")

parser_data.add_argument("-vdset", "--val_datasets", dest="val_datasets", type=str, nargs='+', default=[],
                        choices=["wildtrack", "mot17train", "mot17test", "mot20train", "mot20test", "scouttrain", "scoutval", "scoutmonotrain", "scoutmonoval"],
                        help="Validation datasets. Should match the domain of training data for meaningful validation metrics.")

parser_data.add_argument("-splt", "--split_proportion", dest="split_proportion", type=float, default=0.9,
                        help="Train/validation split ratio (0-1). E.g., 0.9 uses 90%% for training, 10%% for validation.")

parser_data.add_argument("-ftr", "--features", dest="features", type=feature_tuple, action=FeatureDictAction, nargs='+', default=[],
                        help="Node features for the graph. Format: 'type,normalized,hidden_size'. Available: bbox (bounding boxes), world_points (3D coordinates), timestamp, view_id, confidence, reid (re-identification), crops (image patches).")

parser_data.add_argument("-nc", "--num_categories", dest="num_categories", type=int, default=1,
                        help="Number of object categories to track. Currently supports person tracking (nc=1).")

parser_data.add_argument("-cs", "--chunk_size", dest="chunk_size", type=int, default=40,
                        help="Number of consecutive frames processed together. Larger chunks capture longer-term dependencies but use more memory.")

parser_data.add_argument("-ov", "--overlap_size", dest="overlap_size", type=int, default=2,
                        help="Frame overlap between consecutive chunks. Ensures smooth tracking across chunk boundaries.")

parser_data.add_argument("-aug", "--aug_train", dest="aug_train", action="store_true", default=False,
                        help="Enable data augmentation during training. Improves robustness through bbox jittering, scaling, and confidence noise.")

parser_data.add_argument("-vaug", "--views_based_aug_list", dest="views_based_aug_list", type=aug_tuple, nargs='+', default=[(None, 1)],
                        help="View-based augmentations. Format: 'type,probability'. Available: rcrop (random crop), raff (random affine), hflip, vflip.")

parser_data.add_argument("-saug", "--scene_based_aug_list", dest="scene_based_aug_list", type=aug_tuple, nargs='+', default=[(None, 1)],
                        help="Scene-based augmentations applied to entire scenes. Same format as view-based augmentations.")

parser_data.add_argument("-rt", "--reid_type", dest="reid_type", type=str, default="osnet", choices=["osnet"],
                        help="Re-identification model type. OSNet provides robust appearance features for person tracking.")

parser_data.add_argument("-crs", "--crops_size", dest="crops_size", type=int, nargs='+', default=[256, 80],
                        help="Target size for cropped image patches [height, width]. Used for re-identification and appearance features.")

parser_data.add_argument("-mth", "--metric_threshold", dest="metric_threshold", type=float, default=100,
                        help="Metric threshold for tracking metrics. Used for computing tracking metrics. Default is 100 cm for wildtrack. Scout uses meters as based unit so should be set to 1.")

# Detection Configuration
parser_data.add_argument("-udt", "--use_detection_training", dest="use_detection_training", action="store_true", default=False,
                        help="Use automatic detections instead of ground truth bounding boxes during training. Enables end-to-end evaluation.")

parser_data.add_argument("-udv", "--use_detection_eval", dest="use_detection_eval", action="store_true", default=False,
                        help="Use automatic detections during evaluation. Required for real-world deployment scenarios.")

parser_data.add_argument("-dt", "--detection_type", dest="detection_type", type=str, default="yolox",
                        choices=["yolox", "yolox_bytetrack", "yolox_ghost", "rtmdet", "fasterrcnn", "mvaug_frame", "mvaug_ground", "DPM", "SDP", "FRCNN"],
                        help="Object detection model type. YOLOX variants are fastest, RTMDet offers good accuracy-speed trade-off.")

parser_data.add_argument("-dth", "--detection_threshold", dest="detection_threshold", type=float, default=0.7,
                        help="Minimum confidence threshold for detections (0-1). Higher values reduce false positives but may miss difficult detections.")

parser_data.add_argument("-iou", "--iou_threshold_det_gt", dest="iou_threshold_det_gt", type=float, default=0.5,
                        help="IoU threshold for matching detections with ground truth (0-1). Used for generating training supervision.")

parser_data.add_argument("-fmgt", "--fill_missing_gt", dest="fill_missing_gt", action="store_true", default=False,
                        help="Fill missing detections with ground truth bounding boxes during training. Ensures complete supervision.")


####### Model Configuration #######
parser_model.add_argument("-nah", "--num_att_heads", dest="num_att_heads", type=int, default=4,
                         help="Number of attention heads in multi-head attention layers. More heads can capture diverse relationships.")

parser_model.add_argument("-nmp", "--nb_iter_mpnn", dest="nb_iter_mpnn", type=int, default=1,
                         help="Number of message passing iterations. More iterations allow information to propagate further in the graph.")

parser_model.add_argument("-uo", "--use_oracle", dest="use_oracle", action="store_true", default=False,
                         help="Use oracle tracking for evaluation baselines. Provides upper bound performance by using ground truth associations or predicted associations from baseline trackers.")

parser_model.add_argument("-ort", "--oracle_type", dest="oracle_type", type=str, default="graph_edges",
                         choices=["bytetrack", "sort", "bytetrackMV", "pred_id", "graph_edges"],
                         help="Oracle method type. 'graph_edges' uses GT edges, 'bytetrack'/'sort' use classical trackers with GT detections.")

parser_model.add_argument("-dr", "--dropout_rate", dest="dropout_rate", type=float, default=0,
                         help="Dropout rate in MLP layers (0-1). Helps prevent overfitting, especially with small datasets.")

parser_model.add_argument("-cft", "--crop_feature_extractor_type", dest="crop_feature_extractor_type", type=str, default="simple_cnn",
                         choices=["resnet18", "resnet34", "resnet50", "resnet101", "simple_cnn"],
                         help="CNN architecture for extracting features from cropped regions. ResNet variants offer better features but slower inference.")

# Graph Structure Parameters
parser_model.add_argument("-ws", "--window_size", dest="window_size", type=int, default=5,
                         help="Temporal window size for edge creation. Larger windows connect more distant frames but increase computational cost.")

parser_model.add_argument("-tmd", "--temporal_edges_max_time_diff", dest="temporal_edges_max_time_diff", type=int, default=1,
                         help="Maximum frame difference for temporal edges. Controls how far apart in time detections can be connected.")

parser_model.add_argument("-teat", "--temporal_edges_across_views", dest="temporal_edges_across_views", action="store_true", default=False,
                         help="Allow temporal edges between different camera views. Useful for multi-view tracking with synchronized cameras.")

parser_model.add_argument("-vemd", "--view_edges_max_distance", dest="view_edges_max_distance", type=float, default=150,
                         help="Maximum 3D distance (in world units) for creating view edges between detections in different cameras.")

parser_model.add_argument("-tms", "--temporal_edges_max_speed", dest="temporal_edges_max_speed", type=float, default=None,
                         help="Maximum allowed speed for temporal edges (units/frame). Filters unrealistic movement. Use None to disable.")

parser_model.add_argument("-tets", "--temporal_edges_speed_type", dest="temporal_edges_speed_type", type=str, default="world_points",
                         choices=["bbox", "world_points"],
                         help="Coordinate system for speed calculation. 'world_points' more accurate for multi-view, 'bbox' for single-view.")

parser_model.add_argument("-semd", "--social_edges_max_distance", dest="social_edges_max_distance", type=float, default=None,
                         help="Maximum distance for social edges between people in same frame. Models crowd interactions. Use None to disable.")

# Edge Feature Configuration
parser_model.add_argument("-et", "--edge_types", dest="edge_types", type=str, nargs='+', default=["temporal", "view", "social"],
                         choices=["temporal", "view", "social"],
                         help="Types of edges in the graph. 'temporal' connects across time, 'view' across cameras, 'social' within frames.")

parser_model.add_argument("-tef", "--temporal_edges_features", dest="temporal_edges_features", type=feature_tuple, action=FeatureDictAction, nargs='+',
                         default={"zeros": {'is_norm': False, 'hidden_size': 1}},
                         help="Features for temporal edges. Format: 'type,normalized,hidden_size'. Available: zeros, noise, distance, elevation, occlusion, time_distance, bboxiou, bboxratio.")

parser_model.add_argument("-vef", "--view_edges_features", dest="view_edges_features", type=feature_tuple, action=FeatureDictAction, nargs='+',
                         default={"zeros": {'is_norm': False, 'hidden_size': 1}},
                         help="Features for view edges. Same format and options as temporal edge features.")

parser_model.add_argument("-sef", "--social_edges_features", dest="social_edges_features", type=feature_tuple, action=FeatureDictAction, nargs='+',
                         default={"zeros": {'is_norm': False, 'hidden_size': 1}},
                         help="Features for social edges. Same format and options as temporal edge features.")

parser_model.add_argument("-init_edge_agg_type", "--init_edge_agg_type", dest="init_edge_agg_type", type=str, default="mlp",
                         choices=["identity", "mlp"],
                         help="Edge feature aggregation method. 'mlp' learns combinations, 'identity' concatenates features directly.")

# Node Feature Processing
parser_model.add_argument("-dhf", "--dim_hidden_features", dest="dim_hidden_features", type=int, default=64,
                         help="Hidden dimension for node and edge representations in the GNN. Higher values increase model capacity.")

parser_model.add_argument("-init_node_agg_type", "--init_node_agg_type", dest="init_node_agg_type", type=str, default="identity",
                         choices=["identity", "mlp"],
                         help="Node feature aggregation method. 'identity' for direct mapping, 'mlp' for learned transformation.")

# Neural Network Architecture
parser_model.add_argument("-no_res_edge", "--no_residual_edge_update", dest="residual_edge_update", action="store_false", default=True,
                         help="Disable residual connections in edge update networks. May help with gradient flow but usually hurts performance.")

parser_model.add_argument("-no_res_node", "--no_residual_node_update", dest="residual_node_update", action="store_false", default=True,
                         help="Disable residual connections in node update networks. Generally not recommended.")

parser_model.add_argument("-nb_layers_mlp_node", "--nb_layers_mlp_node", dest="nb_layers_mlp_node", type=int, default=2,
                         help="Number of layers in node update MLPs. More layers increase expressiveness but may cause overfitting.")

parser_model.add_argument("-nb_layers_mlp_edge", "--nb_layers_mlp_edge", dest="nb_layers_mlp_edge", type=int, default=2,
                         help="Number of layers in edge update MLPs. Should generally match node MLP depth.")

# Camera Node Configuration (for multi-view with scene modeling)
parser_model.add_argument("-ucn", "--use_camera_node", dest="use_camera_node", action="store_true", default=False,
                         help="Add camera nodes to model scene structure. Useful for multi-view datasets with known camera positions.")

parser_model.add_argument("-cnf", "--camera_node_features", dest="camera_node_features", type=feature_tuple, action=FeatureDictAction, nargs='+',
                         default={"zeros": {'is_norm': False, 'hidden_size': 1}},
                         help="Features for camera nodes. Can include camera parameters, positions, or learned embeddings.")

parser_model.add_argument("-cef", "--camera_edge_features", dest="camera_edge_features", type=feature_tuple, action=FeatureDictAction, nargs='+',
                         default={"zeros": {'is_norm': False, 'hidden_size': 1}},
                         help="Features for edges connecting cameras and detections. Can model occlusions or visibility.")

parser_model.add_argument("-init_camera_agg_type", "--init_camera_agg_type", dest="init_camera_agg_type", type=str, default="mlp",
                         choices=["identity", "mlp"],
                         help="Camera feature aggregation method. 'mlp' usually preferred for camera nodes.")

# Data Augmentation and Regularization
parser_model.add_argument("-dnr", "--drop_node_rate", dest="drop_node_rate", type=float, default=0,
                         help="Node dropout rate during training (0-1). Randomly removes nodes to improve robustness.")

parser_model.add_argument("-der", "--drop_edge_rate", dest="drop_edge_rate", type=float, default=0,
                         help="Edge dropout rate during training (0-1). Randomly removes edges to prevent over-reliance on specific connections.")

parser_model.add_argument("-mnfr", "--mask_node_features_rate", dest="mask_node_features_rate", type=float, default=0,
                         help="Node feature masking rate (0-1). Randomly masks feature dimensions during training.")

parser_model.add_argument("-mefr", "--mask_edge_features_rate", dest="mask_edge_features_rate", type=float, default=0,
                         help="Edge feature masking rate (0-1). Randomly masks edge feature dimensions during training.")

parser_model.add_argument("-mkt", "--masking_type", dest="masking_type", type=str, default="global",
                         choices=["global", "partial"],
                         help="Feature masking strategy. 'global' masks entire features, 'partial' masks individual dimensions.")


####### Loss Configuration #######
parser_loss.add_argument("-lt", "--loss_type", dest="loss_type", type=str, default="focal",
                         choices=["focal", "bce", "l1", "l2"],
                         help="Primary loss function type. 'focal' handles class imbalance better than 'bce' for tracking tasks.")

parser_loss.add_argument("-gamma", "--gamma", dest="gamma", type=float, default=2,
                         help="Focal loss gamma parameter. Higher values focus more on hard examples. Range: 0-5, typically 2.")

parser_loss.add_argument("-alpha", "--alpha", dest="alpha", type=float, default=0.25,
                         help="Focal loss alpha parameter for class balancing. Range: 0-1, typically 0.25.")

parser_loss.add_argument("-sa", "--size_average", dest="size_average", action="store_true", default=True,
                         help="Average loss over batch size. Usually recommended for stable training.")

def get_config_dict(existing_conf_dict=None):
    """
    Generate config dict from command line argument and config file if existing_conf_dict is not None, value are added to the existing dict if they are not already defined, 
    """

    log.debug(f'Original command {" ".join(sys.argv)}')
    args = parser.parse_args()

    if args.config_file is not None:
        yaml_dict = read_yaml_file(args.config_file)
        arg_list = convert_yaml_dict_to_arg_list(yaml_dict)

        args = parser.parse_args(arg_list + sys.argv[1:])

    args_dict = args_to_dict(parser, args)

    config = {"data_conf": {**args_dict["dataset"], **data_path}, "model_conf": {**args_dict["model"], **model_path}, "loss_conf":args_dict["loss"], "training": {**args_dict["training"], **conf_path}, "main":vars(args)}
    
    set_log_level(config["main"]["log_lvl"])
    
    script_name = os.path.basename(sys.argv[0])

    if "eval" in script_name:
        add_log_file_handler(config["training"]["ROOT_PATH"] / "logs" / f"eval_{config['main']['name']}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    else:
        add_log_file_handler(config["training"]["ROOT_PATH"] / "logs" / f"{config['main']['name']}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    config["model_conf"]["edge_features"] = {
        "temporal": args_dict["model"]["temporal_edges_features"],
        "view": args_dict["model"]["view_edges_features"],
        "social": args_dict["model"]["social_edges_features"]
    }

    if existing_conf_dict is not None:
        #Use existing dict as config, and fill it with value if some of them were missing
        config = fill_dict_with_missing_value(existing_conf_dict, config)

    # Check if init_node_agg_type is set to "identity"
    if config["model_conf"]["init_node_agg_type"] == "identity" and len(config["data_conf"]["features"]) > 0:
        # Calculate the sum of all feature dimensions
        total_feature_dim = sum(feature["hidden_size"] for feature in config["data_conf"]["features"].values())
        
        # Set dim_hidden_features to the calculated sum
        config["model_conf"]["dim_hidden_features"] = total_feature_dim
        
        log.info(f"init_node_agg_type is set to 'identity'. Automatically setting dim_hidden_features to {total_feature_dim}")

    return config


if __name__ == '__main__':
    conf_dict = get_config_dict()
