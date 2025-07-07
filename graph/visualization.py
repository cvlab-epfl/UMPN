import torch

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import numpy as np
import cv2

from configs.pathes import data_path
from misc.log_utils import log
from misc.visualization_utils import save_images_as_video


def value_to_rgb(value, cmaps="Reds"):
    value = min(max(value, 0), 1)
    colormap = plt.get_cmap(cmaps)
    return colormap(value)[:3]

def get_color_for_instance(instance_id):
    if instance_id == -1:
        return (0.5, 0.5, 0.5)  # Default color (gray) for instance_id -1
    np.random.seed(instance_id)
    return np.random.rand(3)

def extract_data_and_visualize_graph(graph_data, show_views=None, display_temp_edges=True, display_view_edges=True, display_social_edges=True, start_time=None, end_time=None, pdf_filename=None, only_positive_edges=False):
    # Check if graph_data contains any detections
    if len(graph_data['detection']) == 0:
        log.warning("No data to visualize. Returning empty image.")
        return np.zeros((2000, 2000, 3), dtype=np.uint8)
    
    # Extract necessary data from the graph_data
    world_points = graph_data['detection'].world_points.numpy()
    det_time = graph_data['detection'].timestamp.numpy()
    gt_id = graph_data['detection'].person_id.numpy()
    det_view_id = graph_data['detection'].view_id.numpy()
    
    nb_detection = len(graph_data['detection'].bbox) if hasattr(graph_data['detection'], 'bbox') else 0
    
    edges = {}
    edge_types = ['view', 'temporal', 'social']
    edge_display_flags = [display_view_edges, display_temp_edges, display_social_edges]

    for edge_type, display_flag in zip(edge_types, edge_display_flags):
        if display_flag and ('detection', edge_type, 'detection') in graph_data.edge_types:
            edge_data = graph_data['detection', edge_type, 'detection']
            edge_index = edge_data.edge_index.t().numpy()
            valid_edges = (edge_index[:, 0] < nb_detection) & (edge_index[:, 1] < nb_detection)
            
            if not np.all(valid_edges):
                log.warning(f"Filtered out {np.sum(~valid_edges)} {edge_type} edges pointing to non-existent detections.")
            
            edge_index = edge_index[valid_edges]
            
            if hasattr(edge_data, 'edge_pred'):
                edge_pred = torch.sigmoid(edge_data.edge_pred).numpy()[valid_edges]
            else:
                edge_pred = np.ones((edge_index.shape[0], 1))
            
            if hasattr(edge_data, 'edge_label') and only_positive_edges:
                edge_label = edge_data.edge_label.numpy()[valid_edges]
                labeled_edges = edge_label == 1
                edge_index = edge_index[labeled_edges]
                edge_pred = edge_pred[labeled_edges]
            
            edges[edge_type] = np.column_stack((edge_index, edge_pred))
    
    # If show_views is None, display all views
    if show_views is None:
        show_views = list(np.unique(det_view_id))
    
    # Call the visualization function
    return plot_graph_advanced(world_points[:, :2], det_time, gt_id, det_view_id, 
                                edges.get('view'), edges.get('temporal'), edges.get('social'),
                                edge_threshold=0.5, show_views=show_views, start_time=start_time, end_time=end_time,
                                pdf_filename=pdf_filename, show_gt_edges=False, show_id=False)


def plot_graph_advanced(world_points, det_time, gt_id, det_view_id, view_edges, temporal_edges, social_edges, pred_id=None, edge_threshold=0.5, show_views=[0,1,2,3,4,5,6], show_gt_edges=True, show_id=True, start_time=None, end_time=None, pdf_filename=None):
    fig, ax = plt.subplots(figsize=(20, 20))

    # Create a mask for world_points in show_views and within the specified time range
    mask = np.isin(det_view_id, show_views)
    if start_time is not None:
        mask &= (det_time >= start_time)
    if end_time is not None:
        mask &= (det_time <= end_time)
    
    # Create color map for gt_id
    unique_ids = np.unique(gt_id)
    colors = [get_color_for_instance(int(id)) for id in unique_ids]
    color_map = dict(zip(unique_ids, colors))
    
    # Create custom colormaps
    view_cmap = LinearSegmentedColormap.from_list("", ["white", "red"])
    temporal_cmap = LinearSegmentedColormap.from_list("", ["white", "blue"])
    
    # Plot world_points
    scatter = ax.scatter(world_points[mask, 0], world_points[mask, 1], s=30, c=[color_map[id] for id in gt_id[mask]], alpha=0.7)
    
    # Plot rings for pred_id if available
    if pred_id is not None:
        pred_colors = [get_color_for_instance(int(id)) for id in pred_id[mask]]
        ring_scatter = ax.scatter(world_points[mask, 0], world_points[mask, 1], s=60, facecolors='none', edgecolors=pred_colors, linewidths=2, alpha=0.7)
    
    # Plot ground truth trajectories and add text labels
    if show_gt_edges:
        for id in unique_ids:
            id_mask = (gt_id == id) & mask
            color = color_map[id]
            # Sort the trajectory points by detection time
            sorted_indices = np.argsort(det_time[id_mask])
            ax.plot(world_points[id_mask][sorted_indices, 0], world_points[id_mask][sorted_indices, 1], color=color, alpha=0.6, linewidth=4)
    
    if show_id:
        for i, (point, gt, view) in enumerate(zip(world_points[mask], gt_id[mask], det_view_id[mask])):
            color = color_map[gt]
            label = f"GT: {gt}"
            if pred_id is not None:
                pred = pred_id[mask][i]
                label += f", Pred: {pred}"
            ax.text(point[0], point[1], label, fontsize=8, color=color, ha='left', va='bottom')

    # Plot social edges
    if social_edges is not None:
        social_lines = []
        for src_idx, dst_idx in social_edges:
            if (det_view_id[int(src_idx)] in show_views and 
                det_view_id[int(dst_idx)] in show_views and
                mask[int(src_idx)] and mask[int(dst_idx)]):
                social_lines.append((world_points[int(src_idx)], world_points[int(dst_idx)]))
        
        social_lc = LineCollection(social_lines, colors='pink', alpha=0.3, linewidths=1)
        ax.add_collection(social_lc)

    # Plot temporal edges
    if temporal_edges is not None:
        temporal_lines = []
        temporal_colors = []
        for src_idx, dst_idx, e_score in temporal_edges:
            if (e_score > edge_threshold and 
                det_view_id[int(src_idx)] in show_views and 
                det_view_id[int(dst_idx)] in show_views and
                mask[int(src_idx)] and mask[int(dst_idx)]):
                temporal_lines.append((world_points[int(src_idx)], world_points[int(dst_idx)]))
                temporal_colors.append(value_to_rgb(e_score, "Blues"))
        
        temporal_lc = LineCollection(temporal_lines, colors=temporal_colors, alpha=1, linewidths=1)
        ax.add_collection(temporal_lc)
        
        temporal_sm = plt.cm.ScalarMappable(cmap=temporal_cmap)
        temporal_sm.set_array([])
        cbar_temporal = fig.colorbar(temporal_sm, ax=ax, label="Temporal Edge Score", alpha=1, fraction=0.03, pad=0.04)
    
    # Plot view edges
    if view_edges is not None:
        view_lines = []
        view_colors = []
        for src_idx, dst_idx, e_score in view_edges:
            if (e_score > edge_threshold and 
                det_view_id[int(src_idx)] in show_views and 
                det_view_id[int(dst_idx)] in show_views and
                mask[int(src_idx)] and mask[int(dst_idx)]):
                view_lines.append((world_points[int(src_idx)], world_points[int(dst_idx)]))
                view_colors.append(value_to_rgb(e_score, "Reds"))
        
        view_lc = LineCollection(view_lines, colors=view_colors, alpha=1, linewidths=1)
        ax.add_collection(view_lc)

        view_sm = plt.cm.ScalarMappable(cmap=view_cmap)
        view_sm.set_array([])
        cbar_view = fig.colorbar(view_sm, ax=ax, label="View Edge Score", alpha=1, fraction=0.03, pad=0.04)
    
    # Get the last timestamp
    last_timestamp = np.max(det_time[mask])
    
    # Set the plot limits based on the world points with gt_id != -1
    valid_points_mask = mask & (gt_id != -1)
    x_min, x_max = np.min(world_points[valid_points_mask, 0]), np.max(world_points[valid_points_mask, 0])
    y_min, y_max = np.min(world_points[valid_points_mask, 1]), np.max(world_points[valid_points_mask, 1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.set_title(f"Graph Visualization with Ground Truth and Predicted Trajectories (Last Timestamp: {last_timestamp})")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    if pdf_filename:
        pdf_filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        log.info(f"Graph visualization saved as {pdf_filename}")
        plt.close()
        return None
    else:
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image

def extract_data_and_visualize_single_view(graph_data, selected_view, display_temp_edges=False, display_social_edges=False, pdf_filename=None, last_timestamp=False, frame_image_path=None):
    # Check if graph_data contains any detections
    if len(graph_data['detection']) == 0:
        log.warning("No data to visualize. Returning empty image.")
        return np.zeros((2000, 2000, 3), dtype=np.uint8)        

    # Extract necessary data from the graph_data
    bbox = graph_data['detection'].bbox
    det_time = graph_data['detection'].timestamp
    gt_id = graph_data['detection'].person_id
    det_view_id = graph_data['detection'].view_id

    # Get temporal edges
    temporal_edges = None
    if display_temp_edges and ('detection', 'temporal', 'detection') in graph_data.edge_types:
        temporal_edges = graph_data['detection', 'temporal', 'detection'].edge_index.t().numpy()
        valid_edges = (temporal_edges[:, 0] < len(graph_data['detection'].bbox)) & (temporal_edges[:, 1] < len(graph_data['detection'].bbox))
        filtered_count = np.sum(~valid_edges)
        if filtered_count > 0:
            log.warning(f"Filtered out {filtered_count} temporal edges pointing to non-existent detections.")
        temporal_edges = temporal_edges[valid_edges]

    # Get social edges
    social_edges = None
    if display_social_edges and ('detection', 'social', 'detection') in graph_data.edge_types:
        edge_index = graph_data['detection', 'social', 'detection'].edge_index
        social_edges = edge_index.t().numpy()
        valid_edges = (social_edges[:, 0] < len(graph_data['detection'].bbox)) & (social_edges[:, 1] < len(graph_data['detection'].bbox))
        filtered_count = np.sum(~valid_edges)
        if filtered_count > 0:
            log.warning(f"Filtered out {filtered_count} social edges pointing to non-existent detections.")
        social_edges = social_edges[valid_edges].tolist()

    # Assuming a fixed frame size, you may want to adjust this based on your data
    frame_size = (1920, 1080)  # Example frame size

    return visualize_graph_single_view(
        bbox.cpu().numpy(),
        det_time.cpu().numpy(),
        gt_id.cpu().numpy(),
        det_view_id.cpu().numpy(),
        temporal_edges,
        social_edges,
        frame_size,
        selected_view,
        pdf_filename=pdf_filename,
        last_timestamp=last_timestamp,
        frame_image_path=frame_image_path
    )

def visualize_graph_single_view(bbox, det_time, gt_id, det_view_id, temporal_edges, social_edges, frame_size, selected_view, pdf_filename=None, last_timestamp=False, frame_image_path=None):
    # Filter data for the selected view
    view_mask = det_view_id == selected_view
    bbox_filtered = bbox[view_mask]
    det_time_filtered = det_time[view_mask]
    gt_id_filtered = gt_id[view_mask]

    # Apply last_timestamp filter if specified
    if last_timestamp:
        max_time = np.max(det_time_filtered)
        time_mask = det_time_filtered == max_time
        bbox_filtered = bbox_filtered[time_mask]
        det_time_filtered = det_time_filtered[time_mask]
        gt_id_filtered = gt_id_filtered[time_mask]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Load and display the frame image if provided
    if frame_image_path:
        frame = plt.imread(frame_image_path)
        ax.imshow(frame)
    else:
        ax.set_xlim(0, frame_size[0])
        ax.set_ylim(frame_size[1], 0)  # Invert y-axis to match image coordinates

    # Plot bounding boxes
    for i, box in enumerate(bbox_filtered):
        xmin, ymin, xmax, ymax = box
        color = get_color_for_instance(gt_id_filtered[i])
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Display gt_id and det_time next to the bounding box
        ax.text(xmin, ymin-5, f"ID: {gt_id_filtered[i]}, T: {det_time_filtered[i]}", fontsize=8, color=color)

    # Plot temporal edges
    if temporal_edges is not None and not last_timestamp:
        temp_lines = []
        for src, dst in temporal_edges:
            if view_mask[src] and view_mask[dst]:
                if not last_timestamp or (det_time[src] == det_time[dst] == np.max(det_time_filtered)):
                    src_box = bbox[src]
                    dst_box = bbox[dst]
                    temp_lines.append([(src_box[0] + (src_box[2] - src_box[0])/2, src_box[1] + (src_box[3] - src_box[1])/2),
                                       (dst_box[0] + (dst_box[2] - dst_box[0])/2, dst_box[1] + (dst_box[3] - dst_box[1])/2)])
        temp_lc = LineCollection(temp_lines, colors='g', alpha=0.5, linewidths=1)
        ax.add_collection(temp_lc)

    # Plot social edges
    if social_edges is not None:
        social_lines = []
        for src, dst in social_edges:
            if view_mask[src] and view_mask[dst]:
                if not last_timestamp or (det_time[src] == det_time[dst] == np.max(det_time_filtered)):
                    src_box = bbox[src]
                    dst_box = bbox[dst]
                    social_lines.append([(src_box[0] + (src_box[2] - src_box[0])/2, src_box[1] + (src_box[3] - src_box[1])/2),
                                         (dst_box[0] + (dst_box[2] - dst_box[0])/2, dst_box[1] + (dst_box[3] - dst_box[1])/2)])
        social_lc = LineCollection(social_lines, colors='b', alpha=0.5, linewidths=1)
        ax.add_collection(social_lc)

    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Get the last timestamp value
    last_timestamp_value = np.max(det_time_filtered)

    title = f"Graph Visualization (View {selected_view}) - Last Timestamp: {last_timestamp_value}"
    if last_timestamp:
        title += f" - Last Timestamp Only"
    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    # Remove tight_layout() and use fixed subplot adjustments
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    if pdf_filename:
        plt.savefig(pdf_filename, format='pdf', bbox_inches=None)
        print(f"Graph visualization saved as {pdf_filename}")
        plt.close()
        return None
    else:
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image


def draw_bbox_with_id(frame, bbox, track_id, color, is_pred=True, alpha=0.3):
    """Helper function to draw a single bbox with ID label using cv2"""
    # Convert color from 0-1 RGB to 0-255 BGR
    color_bgr = tuple(int(c * 255) for c in color[::-1])
    
    # Different style for pred vs GT - dashed for pred, solid for GT
    if is_pred:
        # Draw dashed rectangle
        pts = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], 
                       [bbox[2], bbox[3]], [bbox[0], bbox[3]]], np.int32)
        for i in range(4):
            pt1, pt2 = pts[i], pts[(i+1)%4]
            for j in range(0, np.linalg.norm(pt2-pt1).astype(int), 6):
                start = pt1 + (pt2-pt1) * j / np.linalg.norm(pt2-pt1)
                end = pt1 + (pt2-pt1) * min(j+3, np.linalg.norm(pt2-pt1)) / np.linalg.norm(pt2-pt1)
                cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), 
                        color_bgr, 2, cv2.LINE_AA)
    else:
        # Draw solid rectangle
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
                     color_bgr, 2, cv2.LINE_AA)
    
    # Add filled rectangle behind text for better visibility
    text = f'ID: {track_id}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_bg_x1 = int(bbox[0])
    text_bg_y1 = int(bbox[1] - text_h - 4)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (text_bg_x1, text_bg_y1), 
                 (text_bg_x1 + text_w, text_bg_y1 + text_h + 4), 
                 color_bgr, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Add text
    cv2.putText(frame, text, (text_bg_x1, text_bg_y1 + text_h + 2), 
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def visualize_frame(frame, gt_boxes, gt_ids, pred_boxes, pred_ids, show_pred=True, show_gt=True):
    """Helper function to visualize a single frame with predictions and GT"""
    frame = frame.copy()
    
    # Draw predicted boxes
    if show_pred:
        for bbox, pred_id in zip(pred_boxes, pred_ids):
            if pred_id == -1:
                continue
            color = get_color_for_instance(pred_id)
            # Higher alpha (0.8) for unassigned detections (id=-1)
            alpha = 0.8 if pred_id == -1 else 0.3
            draw_bbox_with_id(frame, bbox, pred_id, color, is_pred=True, alpha=alpha)
            
    # Draw ground truth boxes
    if show_gt:
        for bbox, gt_id in zip(gt_boxes, gt_ids):
            color = get_color_for_instance(gt_id)
            draw_bbox_with_id(frame, bbox, gt_id, color, is_pred=False)
    
    return frame

def visualize_sequence_predictions(data, pred_ids, gt_dict, dset_name, sequence, mp4_filename, show_pred, show_gt, selected_views):
    if dset_name == "MOT17":
        type_set = "train" if sequence in ["02", "04", "05", "09", "10", "11", "13"] else "test"  
        frame_template = data_path["MOT17_root"] / type_set / f"MOT17-{sequence}-DPM" / "img1" 
    elif dset_name == "MOT20":
        frame_template = data_path["MOT20_root"] / "train" / f"MOT20-{sequence}" / "img1"
    elif dset_name == "WILDTRACK":
        frame_template = data_path["wildtrack_root"] / "Image_subsets/"
    elif dset_name == "SCOUT":
        frame_template = data_path["scout_root"] / sequence.split("_cam")[0]
    else:
        raise ValueError(f"Dataset {dset_name} not supported")

    for view in selected_views:
        frames = []
        timestamps = sorted(set(data['detection'].timestamp.numpy()))
        
        for t in timestamps:
            # Get frame path
            if dset_name == "wildtrack":
                frame_path = frame_template / f"C{view+1}" / f"{int(t*5):08d}.png"
            elif dset_name == "SCOUT":
                frame_path = frame_template / f"cvlabrpi{view}"
                frame_path = next(frame_path.glob(f"*_{int(t*10)}.jpg"), None)
            else:
                frame_path = frame_template / f"{int(t+1):06d}.jpg"
            
            frame = cv2.imread(str(frame_path))

            if frame is None:
                log.warning(f"Frame not found at {frame_path}")
                continue
            
            # Get detections for this timestamp and view
            mask = (data['detection'].timestamp.numpy() == t) & (data['detection'].view_id.numpy() == view)
            
            pred_boxes = data['detection'].bbox.numpy()[mask]
            curr_pred_ids = pred_ids[mask]
            
            # Get GT for this timestamp and view
            gt_boxes = []
            gt_ids = []
            if t in gt_dict and view in gt_dict[t]:
                gt_boxes = gt_dict[t][view]['bbox']
                gt_ids = gt_dict[t][view]['person_id']
            
            # Visualize frame
            frame_vis = visualize_frame(frame, gt_boxes, gt_ids, pred_boxes, curr_pred_ids, show_pred, show_gt)
            frames.append(frame_vis)
        
        # Save video for this view
        view_mp4_filename = str(mp4_filename).replace('.mp4', f'_view_{view}.mp4')
        save_images_as_video(frames, view_mp4_filename)

# Assuming vertices is a numpy array with shape (n, 5) where columns are [frame, gt_id, view, x, y]
# plot_graph_advanced(vertices[:, 3:], vertices[:, 0], vertices[:, 1], vertices[:, 2], view_edges, temporal_edges, show_views=[0,1,2,3,4,5,6], display_temp_edges=True, display_view_edges=True, start_time=360, end_time=400)