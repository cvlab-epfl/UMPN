import torch

from misc.log_utils import log

def get_graph_stats(graph, print_log=False):
    stats = {}
    
    # Node statistics
    stats['num_detections'] = len(graph['detection'].bbox) if hasattr(graph['detection'], 'bbox') else 0
    stats['num_cameras'] = len(graph['camera'].x) if 'camera' in graph.node_types else 0
    
    total_edges = 0
    # Edge statistics
    for edge_type in ['temporal', 'view', 'social', 'see', 'next']:
        if ('detection', edge_type, 'detection') in graph.edge_types:
            edge_index = graph['detection', edge_type, 'detection'].edge_index
            stats[f'num_{edge_type}_edges'] = edge_index.shape[1]
            total_edges += edge_index.shape[1]
        elif ('camera', edge_type, 'detection') in graph.edge_types:
            edge_index = graph['camera', edge_type, 'detection'].edge_index
            stats[f'num_{edge_type}_edges'] = edge_index.shape[1]
            total_edges += edge_index.shape[1]
        elif ('camera', edge_type, 'camera') in graph.edge_types:
            if hasattr(graph['camera', edge_type, 'camera'], 'edge_index'):
                edge_index = graph['camera', edge_type, 'camera'].edge_index
                stats[f'num_{edge_type}_edges'] = edge_index.shape[1]
                total_edges += edge_index.shape[1]
        else:
            stats[f'num_{edge_type}_edges'] = 0
    
    stats['num_edges'] = total_edges
    
    # Feature statistics
    if hasattr(graph['detection'], 'x') and graph['detection'].x is not None:
        stats['num_detection_features'] = graph['detection'].x.shape[1]
    if 'camera' in graph.node_types and hasattr(graph['camera'], 'x') and graph['camera'].x is not None:
        stats['num_camera_features'] = graph['camera'].x.shape[1]
    
    # Metadata statistics
    for attr in ['timestamp', 'view_id', 'person_id', 'bbox', 'world_points']:
        stats[f'has_{attr}'] = hasattr(graph['detection'], attr)
    
    # Edge attribute statistics
    for edge_type in ['temporal', 'view', 'social', 'see', 'next']:
        for src, dst in [('detection', 'detection'), ('camera', 'detection'), ('camera', 'camera')]:
            if (src, edge_type, dst) in graph.edge_types:
                stats[f'has_{src}_{edge_type}_{dst}_edge_attr'] = hasattr(graph[src, edge_type, dst], 'edge_attr')
                stats[f'has_{src}_{edge_type}_{dst}_edge_label'] = hasattr(graph[src, edge_type, dst], 'edge_label')
                stats[f'has_{src}_{edge_type}_{dst}_edge_pred'] = hasattr(graph[src, edge_type, dst], 'edge_pred')
    
    # Time window statistics
    if hasattr(graph['detection'], 'timestamp') and len(graph['detection'].timestamp) > 0:
        timestamps = graph['detection'].timestamp
        stats['time_window'] = timestamps.max() - timestamps.min()
    
    # View statistics
    if hasattr(graph['detection'], 'view_id'):
        stats['num_views'] = len(torch.unique(graph['detection'].view_id))
    
    if print_log:
        # Log the stats
        log.info(f"Graph statistics:")
        for key, value in stats.items():
            log.info(f"  {key}: {value}")
    
    return stats

def compute_graph_cost_from_id(data, pred_ids, min_prob=1e-15):
    """
    Compute graph cost based on predicted IDs by summing negative log likelihood 
    of edges connecting detections with same ID.

    Args:
        data: HeteroData graph containing detection nodes and edges
        pred_ids: Tensor of predicted IDs for each detection

    Returns:
        Total cost (negative log likelihood) of edges between same-ID detections
    """
    # Get temporal and view edge indices and predictions
    temporal_edge_index = data[('detection', 'temporal', 'detection')].edge_index
    view_edge_index = data[('detection', 'view', 'detection')].edge_index
    
    temporal_edge_pred = data[('detection', 'temporal', 'detection')].edge_pred
    view_edge_pred = data[('detection', 'view', 'detection')].edge_pred

    # Convert predictions to probabilities using sigmoid
    temporal_probs = torch.clamp(torch.sigmoid(temporal_edge_pred), min=min_prob, max=1-min_prob)
    view_probs = torch.clamp(torch.sigmoid(view_edge_pred), min=min_prob, max=1-min_prob)

    # Find edges connecting same ID detections and filter out -1 IDs
    temporal_valid = (pred_ids[temporal_edge_index[0]] != -1) | (pred_ids[temporal_edge_index[1]] != -1)
    view_valid = (pred_ids[view_edge_index[0]] != -1) | (pred_ids[view_edge_index[1]] != -1)
    
    temporal_same_id = (pred_ids[temporal_edge_index[0]] == pred_ids[temporal_edge_index[1]]) & temporal_valid
    view_same_id = (pred_ids[view_edge_index[0]] == pred_ids[view_edge_index[1]]) & view_valid

    # Compute negative log likelihood, but only for edges connecting same valid ID
    temporal_cost = -torch.log(temporal_probs[temporal_same_id] / torch.clamp(1 - temporal_probs[temporal_same_id], min=min_prob)).sum()
    view_cost = -torch.log(view_probs[view_same_id] / torch.clamp(1 - view_probs[view_same_id], min=min_prob)).sum()

    # Total cost is sum of temporal and view costs
    total_cost = temporal_cost + view_cost

    return total_cost