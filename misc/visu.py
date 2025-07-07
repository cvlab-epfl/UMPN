from matplotlib import pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

def value_to_rgb(value, cmaps="Reds"):
    value = min(max(value, 0), 1)
    colormap = plt.get_cmap(cmaps)
    return colormap(value)[:3]

def get_color_for_instance(instance_id):
    np.random.seed(instance_id)
    return np.random.rand(3)

def plot_graph(detections, det_time, gt_id, det_view_id, view_edges, temporal_edges, edge_threshold=0.5, show_views=[0,1,2,3,4,5,6], display_temp_edges=False, display_view_edges=False, start_time=None, end_time=None):
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Create a mask for detections in show_views and within the specified time range
    mask = np.isin(det_view_id, show_views)
    if start_time is not None:
        mask &= (det_time >= start_time)
    if end_time is not None:
        mask &= (det_time <= end_time)
    
    # Create color map for gt_id
    unique_ids = np.unique(gt_id)
    colors = [get_color_for_instance(int(id)) for id in unique_ids]
    color_map = dict(zip(unique_ids, colors))
    
    # Plot detections
    scatter = ax.scatter(detections[mask, 0], detections[mask, 1], s=30, c=[color_map[id] for id in gt_id[mask]], alpha=0.7)
    
    # Plot ground truth trajectories
    for id in unique_ids:
        id_mask = (gt_id == id) & mask
        color = color_map[id]
        # Sort the trajectory points by detection time
        sorted_indices = np.argsort(det_time[id_mask])
        ax.plot(detections[id_mask][sorted_indices, 0], detections[id_mask][sorted_indices, 1], color=color, alpha=0.3, linewidth=2)
    
    # Create custom colormaps
    view_cmap = LinearSegmentedColormap.from_list("", ["white", "red"])
    temporal_cmap = LinearSegmentedColormap.from_list("", ["white", "blue"])
    
    # Plot view edges
    if display_view_edges:
        view_lines = []
        view_colors = []
        for src_idx, dst_idx, e_score in view_edges:
            if (e_score > edge_threshold and 
                det_view_id[int(src_idx)] in show_views and 
                det_view_id[int(dst_idx)] in show_views and
                mask[int(src_idx)] and mask[int(dst_idx)]):
                view_lines.append((detections[int(src_idx)], detections[int(dst_idx)]))
                view_colors.append(value_to_rgb(e_score, "Reds"))
        
        view_lc = LineCollection(view_lines, colors=view_colors, alpha=0.5, linewidths=1)
        ax.add_collection(view_lc)

        view_sm = plt.cm.ScalarMappable(cmap=view_cmap)
        view_sm.set_array([])
        cbar_view = fig.colorbar(view_sm, ax=ax, label="View Edge Score", alpha=0.5, fraction=0.03, pad=0.04)
    
    # Plot temporal edges
    if display_temp_edges:
        temporal_lines = []
        temporal_colors = []
        for src_idx, dst_idx, e_score in temporal_edges:
            if (e_score > edge_threshold and 
                det_view_id[int(src_idx)] in show_views and 
                det_view_id[int(dst_idx)] in show_views and
                mask[int(src_idx)] and mask[int(dst_idx)]):
                temporal_lines.append((detections[int(src_idx)], detections[int(dst_idx)]))
                temporal_colors.append(value_to_rgb(e_score, "Blues"))
        
        temporal_lc = LineCollection(temporal_lines, colors=temporal_colors, alpha=0.5, linewidths=1)
        ax.add_collection(temporal_lc)
        
        temporal_sm = plt.cm.ScalarMappable(cmap=temporal_cmap)
        temporal_sm.set_array([])
        cbar_temporal = fig.colorbar(temporal_sm, ax=ax, label="Temporal Edge Score", alpha=0.5, fraction=0.03, pad=0.04)
    
    ax.set_title("Graph Visualization with Ground Truth Trajectories")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

# Assuming vertices is a numpy array with shape (n, 5) where columns are [frame, gt_id, view, x, y]
plot_graph(vertices[:, 3:], vertices[:, 0], vertices[:, 1], vertices[:, 2], view_edges, temporal_edges, show_views=[0,1,2,3,4,5,6], display_temp_edges=True, display_view_edges=True, start_time=0, end_time=100)
