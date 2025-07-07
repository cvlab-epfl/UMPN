import numpy as np
import trimesh
import torch
import time
import os
from pathlib import Path

from configs.pathes import data_path
from misc.log_utils import log
from misc.geometry import load_mesh

# Assuming there's a mesh file path defined in pathes.py
if 'scout_root' in data_path:
    mesh_path = data_path['scout_root'] / "meshes" / "mesh.ply"
    scene_mesh = load_mesh(mesh_path)
else:
    log.error("Mesh file path not defined in data_path")
    scene_mesh = None


def get_segment_mesh_intersections(point1, point2):
    """
    Find intersections between a line segment and the scene mesh.
    
    Args:
        point1 (array-like): First 3D point of the segment
        point2 (array-like): Second 3D point of the segment
        
    Returns:
        tuple: (
            intersections: list of points where segment intersects mesh,
            distances: distances from point1 to each intersection,
            count: total number of intersections
        )
    """
    if scene_mesh is None:
        return [], [], 0
    
    # Convert to numpy arrays if not already
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Calculate direction vector
    direction = point2 - point1
    length = np.linalg.norm(direction)
    
    if length < 1e-6:  # If points are too close
        return [], [], 0
    
    # Normalize direction
    direction = direction / length
    
    # Use trimesh's ray.intersects_location to find intersections
    # We use a ray from point1 in the direction of point2, but limit to the segment length
    locations, index_ray, index_tri = scene_mesh.ray.intersects_location(
        ray_origins=[point1],
        ray_directions=[direction],
        multiple_hits=True
    )
    
    if len(locations) == 0:
        return [], [], 0
    
    # Calculate distances from point1 to each intersection
    distances = []
    valid_intersections = []
    
    for loc in locations:
        # Distance from point1 to intersection
        dist = np.linalg.norm(loc - point1)
        
        # Only consider intersections along the segment (not beyond point2)
        if dist <= length:
            distances.append(dist)
            valid_intersections.append(loc)
    
    # Sort intersections by distance
    sorted_idx = np.argsort(distances)
    sorted_distances = [distances[i] for i in sorted_idx]
    sorted_intersections = [valid_intersections[i] for i in sorted_idx]
    
    return sorted_intersections, sorted_distances, len(sorted_intersections)

def is_occluded(point1, point2, threshold=1):
    """
    Check if line of sight between two points is occluded by the mesh.
    
    Args:
        point1 (array-like): First 3D point
        point2 (array-like): Second 3D point
        threshold (int): Number of intersections considered as occlusion
        
    Returns:
        bool: True if occluded, False otherwise
    """
    _, _, count = get_segment_mesh_intersections(point1, point2)
    return count >= threshold

def occlusion_ratio(point1, point2):
    """
    Calculate the occlusion ratio between two points based on segment length and intersection count.
    
    Args:
        point1 (array-like): First 3D point
        point2 (array-like): Second 3D point
        
    Returns:
        float: Ratio of occlusion (0.0 = no occlusion, 1.0 = fully occluded)
    """
    _, distances, count = get_segment_mesh_intersections(point1, point2)
    
    if count == 0:
        return 0.0
        
    total_distance = np.linalg.norm(np.array(point2) - np.array(point1))
    
    # A simple heuristic: more intersections = more occlusion
    # Can be made more sophisticated based on specific needs
    return min(1.0, count / 3.0)  # Assuming 3+ intersections means fully occluded

def occlusion_score(point1, point2):
    """
    Calculate a continuous occlusion score between two points.
    
    Args:
        point1 (array-like): First 3D point
        point2 (array-like): Second 3D point
        
    Returns:
        float: Occlusion score between 0.0 (completely visible) and 1.0 (completely occluded)
    """
    _, distances, count = get_segment_mesh_intersections(point1, point2)
    
    if count == 0:
        return 0.0
    
    # Calculate total segment length
    total_length = np.linalg.norm(np.array(point2) - np.array(point1))
    
    # Calculate a score based on both count and distances
    # The closer the first intersection is to point1, the higher the occlusion
    if len(distances) > 0:
        first_hit_ratio = distances[0] / total_length
        return 1.0 - first_hit_ratio * (0.7 ** (count - 1))
    
    return 0.0

def camera_person_occlusion(camera_position, person_feet_position, person_height=1.75):
    """
    Calculate occlusion between a camera and a person.
    
    Args:
        camera_position (array-like): 3D coordinates of the camera
        person_feet_position (array-like): 3D coordinates at the feet of the person
        person_height (float, optional): Height of the person in meters. Defaults to 1.75.
        
    Returns:
        torch.Tensor: Tensor of shape (3,) containing occlusion scores for feet, mid-body, and head
                     with values between 0.0 (completely visible) and 1.0 (completely occluded)
    """
    # Convert inputs to numpy arrays for calculations
    camera_position = np.array(camera_position)
    person_feet_position = np.array(person_feet_position)
    
    # Calculate the position of the person's head and mid-body
    person_head_position = person_feet_position.copy()
    person_head_position[2] += person_height
    
    person_mid_position = (person_feet_position + person_head_position) / 2
    
    # Check occlusion for different parts of the body
    feet_occlusion = occlusion_score(camera_position, person_feet_position)
    mid_occlusion = occlusion_score(camera_position, person_mid_position)
    head_occlusion = occlusion_score(camera_position, person_head_position)
    

    return torch.tensor([feet_occlusion, mid_occlusion, head_occlusion], dtype=torch.float)
