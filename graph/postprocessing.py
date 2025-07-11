import torch
import json

from misc.log_utils import log


def extract_tracjectory_with_tracker(graph, tracker_type='bytetrack'):
    
    
    # Extract detection data from graph
    detections = graph['detection']
    bboxes = detections.bbox
    
    timestamps = detections.timestamp
    confidences = detections.confidence

    if tracker_type == 'bytetrack':
        from tracker.tracker import run_bytetrack_on_detections
        tracking_results = run_bytetrack_on_detections(
            bboxes=bboxes,
            timestamps=timestamps, 
            confidences=confidences,
            track_on_ground=False,
            use_motion=False,
            use_kalman_filter=True,
            tracker_interval=1
        )
    elif tracker_type == 'sort':
        from tracker.tracker import run_sort_on_detections
        tracking_results = run_sort_on_detections(
            bboxes=bboxes,
            timestamps=timestamps, 
            confidences=confidences,
            track_on_ground=False,
            use_motion=False,
            use_kalman_filter=True,
            tracker_interval=1
        )
    elif tracker_type == 'bytetrackMV':
        world_points = detections.world_points
        view_ids = detections.view_id

        from tracker.tracker import run_bytetrack3d_on_detections
        tracking_results = run_bytetrack3d_on_detections(
            bboxes=bboxes,
            world_points=world_points,
            view_ids=view_ids,
            timestamps=timestamps, 
            confidences=confidences,
            use_kalman_filter=True,
            tracker_interval=1
        )
    else:
        raise ValueError(f"Invalid tracker type: {tracker_type}")

    # Convert tracking results to trajectory IDs
    num_detections = len(detections.timestamp)
    trajectory_ids = torch.full((num_detections,), -1, dtype=torch.long)
    
    unique_timestamps = torch.unique(timestamps, sorted=True)

    # Assign trajectory IDs based on tracking results
    for frame_result, timestamp in zip(tracking_results, unique_timestamps):
        frame_bboxes = frame_result["bboxes"]
        frame_ids = frame_result["instances_id"]

        
        match_nb_idx = -1
        
        # Find matching detections in original graph
        for track_bbox, track_id in zip(frame_bboxes, frame_ids):
            # Find detection index that matches this bbox and timestamp
            def box_iou(box1, boxes):
                # Convert single box to [1,4] if needed
                if box1.dim() == 1:
                    box1 = box1.unsqueeze(0)
                
                # Calculate areas
                area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
                area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                
                # Calculate intersection
                lt = torch.max(box1[:, :2], boxes[:, :2])  # Left-top
                rb = torch.min(box1[:, 2:], boxes[:, 2:])  # Right-bottom
                
                wh = (rb - lt).clamp(min=0)  # Width-height
                inter = wh[:, 0] * wh[:, 1]  # Intersection area
                
                # Calculate union
                union = area1 + area2 - inter
                
                # Calculate IoU
                iou = inter / union
                return iou
                
            ious = box_iou(track_bbox, bboxes)
            bbox_matches = ious > 0.95
            time_matches = timestamps == timestamp
            matches = bbox_matches & time_matches
            multi_match = 0

            if matches.sum() > 1:
                log.warning(f"Multiple detections matched to the same track for timestamp {timestamp}")
                multi_match = 1
                match_nb_idx += 1

            # assert matches.sum() == 1, "Multiple detections matched to the same track"
            
            if matches.any():
                match_idx = torch.where(matches)[0]
                det_idx = match_idx[min(multi_match*match_nb_idx, match_idx.shape[0]-1)]
                trajectory_ids[det_idx] = track_id

    # Group detections by trajectory ID
    trajectories = {}
    for i, tid in enumerate(trajectory_ids):
        if tid == -1:
            continue
        if tid not in trajectories:
            trajectories[tid] = []
        trajectories[tid].append(i)

    trajectory_list = list(trajectories.values())

    return trajectory_list, trajectory_ids

def extract_oracles_trajectories_pred_id(graph):
    detections = graph['detection']
    num_detections = len(detections.timestamp)
    person_ids = detections.person_id

    # Group detections by person_id
    trajectories = {}
    trajectory_ids = [-1] * num_detections
    next_id = 1

    # Iterate through all detections
    for i in range(num_detections):
        pid = person_ids[i].item()
        # Skip detections with invalid person_id (-1)
        if pid == -1:
            continue
            
        if pid not in trajectories:
            trajectories[pid] = []
        trajectories[pid].append(i)

    # Convert trajectories dict to list, filtering out single detections
    trajectory_list = [traj for traj in trajectories.values() if len(traj) > 1]

    # Assign trajectory IDs
    for i, traj in enumerate(trajectory_list):
        for det_idx in traj:
            trajectory_ids[det_idx] = i + 1  # Start IDs from 1

    trajectory_ids = torch.tensor(trajectory_ids)

    return trajectory_list, trajectory_ids

def extract_oracles_trajectories(graph):
    detections = graph['detection']
    num_detections = len(detections.timestamp)

    adjacency = [[] for _ in range(num_detections)]
    edge_types = ['temporal', 'view']
    for edge_type in edge_types:
        if ('detection', edge_type, 'detection') in graph.edge_types:
            edges = graph['detection', edge_type, 'detection']
            edge_index = edges.edge_index
            edge_label = edges.edge_label.squeeze()
            
            # Filter edges based on ground truth label
            valid_edges = edge_index[:, edge_label == 1]
            
            # Build adjacency list
            for src, dst in valid_edges.t().tolist():
                adjacency[src].append(dst)
                adjacency[dst].append(src)  # Undirected graph
    
    # Union-Find for trajectory extraction
    parent = list(range(num_detections))
    rank = [0] * num_detections

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            parent[px] = py
        elif rank[px] > rank[py]:
            parent[py] = px
        else:
            parent[py] = px
            rank[px] += 1

    # Perform union operations
    for i in range(num_detections):
        for j in adjacency[i]:
            union(i, j)

    # Extract trajectories
    trajectories = {}
    for i in range(num_detections):
        root = find(i)
        if root not in trajectories:
            trajectories[root] = []
        trajectories[root].append(i)

    # Convert to list of trajectories, excluding lone detections
    trajectory_list = [traj for traj in trajectories.values() if len(traj) > 1]

    # Assign trajectory IDs
    trajectory_ids = [-1] * num_detections
    for i, traj in enumerate(trajectory_list):
        for det_idx in traj:
            trajectory_ids[det_idx] = i + 1  # Start IDs from 1
    
    trajectory_ids = torch.tensor(trajectory_ids)

    return trajectory_list, trajectory_ids
            
def extract_multiview_trajectories(graph, edge_threshold=0.5, node_threshold=0.5, min_trajectory_length=5):
        
    detections = graph['detection']
    num_detections = len(detections.timestamp)

    adjacency = [[] for _ in range(num_detections)]
    
    # Filter out nodes below the threshold
    node_pred = torch.sigmoid(detections.pred).squeeze()
    valid_nodes = node_pred > node_threshold
    
    edge_types = ['temporal', 'view']
    adjacency_short = [[] for _ in range(num_detections)]  # For edges with time_diff <= 1
    adjacency_long = [[] for _ in range(num_detections)]   # For edges with time_diff > 1
    
    for edge_type in edge_types:
        if ('detection', edge_type, 'detection') in graph.edge_types:
            edges = graph['detection', edge_type, 'detection']
            edge_index = edges.edge_index
            edge_pred = torch.sigmoid(edges.edge_pred).squeeze()
            
            # Filter edges based on threshold and valid nodes
            high_prob_mask = (edge_pred > edge_threshold) & valid_nodes[edge_index[0]] & valid_nodes[edge_index[1]]
            high_prob_edges = edge_index[:, high_prob_mask]
            high_prob_edges_score = edge_pred[high_prob_mask]

            # Filter edges based on timestamp difference
            timestamps = detections.timestamp
            time_diffs = torch.abs(timestamps[high_prob_edges[0]] - timestamps[high_prob_edges[1]])
            
            # Split edges into short and long time differences
            short_mask = time_diffs <= 1
            long_mask = time_diffs > 1
            
            # Process short-term edges
            short_edges = high_prob_edges[:, short_mask]
            short_scores = high_prob_edges_score[short_mask]
            for (src, dst), score in zip(short_edges.t().tolist(), short_scores):
                adjacency_short[src].append((dst, score))
                adjacency_short[dst].append((src, score))
                
            # Process long-term edges 
            long_edges = high_prob_edges[:, long_mask]
            long_scores = high_prob_edges_score[long_mask]

            # Filter long edges with a stricter threshold
            long_edge_threshold = edge_threshold * 1  # 50% higher threshold for long edges
            stricter_mask = high_prob_edges_score[long_mask] > long_edge_threshold
            long_edges = long_edges[:, stricter_mask]
            long_scores = long_scores[stricter_mask]

            for (src, dst), score in zip(long_edges.t().tolist(), long_scores):
                adjacency_long[src].append((dst, score))
                adjacency_long[dst].append((src, score))
                
    # Sort both adjacency lists by score
    for i in range(num_detections):
        adjacency_short[i].sort(key=lambda x: x[1], reverse=True)
        adjacency_short[i] = [neighbor for neighbor, _ in adjacency_short[i]]
        
        adjacency_long[i].sort(key=lambda x: x[1], reverse=True)
        adjacency_long[i] = [neighbor for neighbor, _ in adjacency_long[i]]
        
    # Use short-term adjacency as primary adjacency
    adjacency = adjacency_short
        
    # Union-Find with conflict checking
    parent = [i for i in range(num_detections)]
    component_metadata = [{} for _ in range(num_detections)]  # Each component's (timestamp, view_id) pairs

    # Initialize component metadata
    for i in range(num_detections):
        timestamp = detections.timestamp[i].item()
        view_id = detections.view_id[i].item()
        component_metadata[i] = {(timestamp, view_id)}

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])  # Path compression
        return parent[u]

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu == pv:
            return
        # Check for conflicts and valid nodes
        if valid_nodes[u] and valid_nodes[v]:
            metadata_u = component_metadata[pu]
            metadata_v = component_metadata[pv]
            if metadata_u.isdisjoint(metadata_v):
                # Merge components
                parent[pu] = pv
                # Merge metadata
                component_metadata[pv] = metadata_u.union(metadata_v)
        else:
            # Do not merge if either node is invalid
            pass

    # Perform Union-Find with conflict checking
    if False:
        for u in range(num_detections):
            for v in adjacency[u]:
                union(u, v)

        for u in range(num_detections):
            for v in adjacency_long[u]:
                union(u, v)
    elif True:
        # Process windows sequentially
        window_size = 10  # Using the argument passed to the function
        current_window_start = detections.timestamp[0]
        
        while current_window_start <= detections.timestamp[-1]:
            window_end = current_window_start + window_size
            
            # First process normal adjacency within window
            for u in range(num_detections):
                if current_window_start <= detections.timestamp[u] < window_end:
                    for v in adjacency[u]:
                            union(u, v)
            
            # Then process long adjacency from previous window
            for u in range(num_detections):
                if current_window_start - window_size <= detections.timestamp[u] < current_window_start:
                    for v in adjacency_long[u]:
                            union(u, v)
            
            current_window_start += window_size  # Slide window by 1 timestamp
    else:
        while any(adjacency):
            for u in range(num_detections):
                if adjacency[u]:
                    v = adjacency[u].pop(0)  # Get the first element
                    union(u, v)

    # Assign trajectory IDs based on connected components
    trajectory_dict = {}
    for i in range(num_detections):
        if valid_nodes[i]:
            traj_id = find(i)
            if traj_id not in trajectory_dict:
                trajectory_dict[traj_id] = []
            trajectory_dict[traj_id].append(i)

    # Remove trajectories shorter than min_trajectory_length
    trajectory_dict = {k: v for k, v in trajectory_dict.items() if len(v) >= min_trajectory_length}

    # Renumber trajectory IDs to be sequential
    trajectories = {}
    pred_id = [-1] * num_detections
    for new_id, (old_id, indices) in enumerate(trajectory_dict.items()):
        trajectories[new_id] = indices
        for idx in indices:
            pred_id[idx] = new_id
    
    pred_id = torch.tensor(pred_id)

    return trajectories, pred_id


def export_graph_as_json(graph, json_path):
    """
    Export the graph data as a JSON file.

    Args:
        json_path (str): Path to save the JSON file.
        use_archived (bool): Whether to use archived data instead of current data.

    The exported JSON will have the following structure:
    {
        "vertices": List[List[float]], # Shape: (N, 5) - [Frame_id, Gt_person_id, view_id, X, Y]
        "vertices_scores": List[float], # Shape: (N,) - vertice score, -1 if vertice not active in that window
        "temporal_edges": List[List[float]], # Shape: (M, 3) - [Src_vertex, Dst_vertex, edge_score]
        "view_edges": List[List[float]] # Shape: (K, 3) - [Src_vertex, Dst_vertex, edge_score]
    }
    Where N is the number of vertices, M is the number of temporal edges, and K is the number of view edges.
    """
    
    export_data = {
        "vertices": [],
        "vertices_scores": [],
        "temporal_edges": [],
        "view_edges": []
    }

    # Export vertices
    if hasattr(graph['detection'], 'timestamp') and hasattr(graph['detection'], 'person_id') and \
        hasattr(graph['detection'], 'view_id') and hasattr(graph['detection'], 'world_points'):
        export_data["vertices"] = torch.stack([
            graph['detection'].timestamp,
            graph['detection'].person_id,
            graph['detection'].view_id,
            graph['detection'].world_points[:, 0],
            graph['detection'].world_points[:, 1]
        ], dim=1).tolist()

    # Export vertices scores
    if hasattr(graph['detection'], 'pred'):
        export_data["vertices_scores"] = torch.sigmoid(graph['detection'].pred).tolist()

    # Export bounding boxes
    if hasattr(graph['detection'], 'bbox'):
        export_data["bbox"] = graph['detection'].bbox.tolist()

    # Export temporal edges
    if ('detection', 'temporal', 'detection') in graph.edge_types:
        temporal_edges = graph['detection', 'temporal', 'detection']
        export_data["temporal_edges"] = torch.cat([
            temporal_edges.edge_index.t(),
            torch.sigmoid(temporal_edges.edge_pred)
        ], dim=1).tolist()

    # Export view edges
    if ('detection', 'view', 'detection') in graph.edge_types:
        view_edges = graph['detection', 'view', 'detection']
        export_data["view_edges"] = torch.cat([
            view_edges.edge_index.t(),
            torch.sigmoid(view_edges.edge_pred)
        ], dim=1).tolist()

    # Create parent directory if it doesn't exist
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Save to JSON file
    with open(json_path, 'w') as f:
        json.dump(export_data, f)

    log.info(f"Graph exported to {json_path}")