import cv2
import numpy as np
import torch 
import trimesh

from scipy.interpolate import interp1d
from skimage.draw import polygon, polygon_perimeter
from sympy.geometry.util import intersection, convex_hull
from sympy import Point, Polygon

from misc.log_utils import log

def project_to_ground_plane_pytorch(img, H, homography_input_size, homography_output_size, grounplane_img_size, padding_mode="zeros"):

    if len(img.shape) == 5:
        Bor, V = img.shape[:2]
        img = img.view(-1, *img.shape[2:])
        H = H.view(-1, *H.shape[2:])
    else:
        V = -1

    if list(img.shape[-2:]) != homography_input_size:
        #TODO resize
        # assert (np.array(org_input_size) != np.array(img.shape[-2:])).all()
        # #interpolate image to original sizz to make groundview projection correct
        log.spam("resising before projection")
        img = torch.nn.functional.interpolate(img, size=tuple(homography_input_size))

    #For proper alignement grounplane_img_size must match the value use to generate the homography
    B, C, h, w = img.shape

    h_grid, w_grid = homography_output_size
  
    yp_dist, xp_dist = torch.meshgrid(torch.arange(h_grid, device=img.device), torch.arange(w_grid, device=img.device))
    homogenous = torch.stack([xp_dist.float(), yp_dist.float(), torch.ones((h_grid, w_grid), device=img.device)]).reshape(1, 3, -1)
    
    if B != 1:
        homogenous = homogenous.repeat(B, 1, 1)

    map_ind  = H.bmm(homogenous)
    
    map_ind = (map_ind[:, :-1, :]/map_ind[:, -1, :].unsqueeze(1)).reshape(B, 2, h_grid, w_grid)
    map_ind = (map_ind / torch.tensor([w-1, h-1], device=img.device).reshape(1,2,1,1))*2 - 1

    grid = map_ind.permute(0,2,3,1)
    
    if padding_mode=="border":
        #Set the border to min value before projection and use min border for padding
        min_val = img.view(B, -1).min(dim=1).values.unsqueeze(1).unsqueeze(1)

        img[:,:,0,:] = min_val
        img[:,:,:,0] = min_val
        img[:,:,h-1,:] = min_val
        img[:,:,:,w-1] = min_val

    ground_image = torch.nn.functional.grid_sample(img, grid, mode='bilinear', align_corners=True,  padding_mode=padding_mode)

    ground_image[torch.isnan(ground_image)] = 0

    if grounplane_img_size != homography_output_size:
        #TODO resize
        log.spam("resising after projection")
        ground_image = torch.nn.functional.interpolate(ground_image, size=tuple(grounplane_img_size))

    if V != -1:
        ground_image = ground_image.view(Bor,V, *ground_image.shape[1:])
    # ground_image = ground_image_torch.squeeze().permute(1,2,0).numpy()

    return ground_image

# def project_to_ground_plane_pytorch(img, H, homography_input_size, homography_output_size, grounplane_img_size, padding_mode="zeros"):

#     if list(img.shape[-2:]) != homography_input_size:
#         #TODO resize
#         # assert (np.array(org_input_size) != np.array(img.shape[-2:])).all()
#         # #interpolate image to original sizz to make groundview projection correct
#         log.debug("resising before projection")
#         img = torch.nn.functional.interpolate(img, size=tuple(homography_input_size))


#     #For proper alignement grounplane_img_size must match the value use to generate the homography
#     if len(img.shape) == 5:
#         B, V, C, h, w = img.shape
#     else:
#         B, C, h, w = img.shape
#         V = 1

#     h_grid, w_grid = homography_output_size
  
#     yp_dist, xp_dist = torch.meshgrid(torch.arange(w_grid, device=img.device), torch.arange(h_grid, device=img.device))
#     homogenous = torch.stack([xp_dist.float(), yp_dist.float(), torch.ones((w_grid, h_grid), device=img.device)]).reshape(1, 3, -1)
    
#     if V == 1:
#         homogenous = homogenous.repeat(B, 1, 1)
#     else:
#         homogenous = homogenous.repeat(B, V, 1, 1)
    
#     H = H.view(B*V,3,3)
#     homogenous = homogenous.view(B*V,3,-1)

#     map_ind  = H.bmm(homogenous)
    
#     map_ind = (map_ind[:, :-1, :]/map_ind[:, -1, :].unsqueeze(1)).reshape(B*V, 2, w_grid, h_grid)
#     map_ind = (map_ind / torch.tensor([w-1, h-1], device=img.device).reshape(1,2,1,1))*2 - 1
    
#     grid = map_ind.permute(0,2,3,1)
    
#     if padding_mode=="border":
#         #Set the border to min value before projection and use min border for padding
#         min_val = img.view(B, -1).min(dim=1).values.unsqueeze(1).unsqueeze(1)

#         img[:,:,0,:] = min_val
#         img[:,:,:,0] = min_val
#         img[:,:,h-1,:] = min_val
#         img[:,:,:,w-1] = min_val

#     if V != 1:
#         img = img.view(B*V,C,h,w)

#     ground_image = torch.nn.functional.grid_sample(img, grid, align_corners=True,  padding_mode=padding_mode)

#     ground_image[torch.isnan(ground_image)] = 0

#     if V != 1:
#         ground_image = ground_image.view(B,V,C,grounplane_img_size[0],grounplane_img_size[1])

#     if grounplane_img_size != homography_output_size:
#         #TODO resize
#         log.debug("resising after projection")
#         ground_image = torch.nn.functional.interpolate(ground_image, size=tuple(grounplane_img_size))

#     # ground_image = ground_image_torch.squeeze().permute(1,2,0).numpy()

#     return ground_image


def get_ground_plane_homography(K, R, T, world_origin_shift, grounplane_img_size, scale, height=0, grounplane_size_for_scale=None):

    #the if the scale is set with respect to particular grounplane size (grounplane_size_for_scale) and the homography has different ouput dimension
    #the scale is adjust such that the ouput is consistent no matter grounplane_img_size
    #in most cases scalex and scaley should be the same
    if grounplane_size_for_scale is not None and grounplane_img_size != grounplane_size_for_scale:
        scalex = scale * (grounplane_img_size[1] / grounplane_size_for_scale[1])
        scaley = scale * (grounplane_img_size[0] / grounplane_size_for_scale[0])
    else:
        scalex = scale
        scaley = scale

    Ki = np.array([[-scalex, 0, ((grounplane_img_size[1]-1)/2)], [0, scaley, ((grounplane_img_size[0]-1)/2)], [0, 0, 1]])

    T =  T + (R @ np.array([[world_origin_shift[0], world_origin_shift[1], height]]).T)
    
    RT = np.zeros((3,3))
    RT[:,:2] = R[:,:2]
    # RT[2,2] = 1
    RT[:,2] = T.squeeze()
    
    H = K @ RT @ np.linalg.inv(Ki)
    
    return H

def get_homograhy_from_corner_points(img_corner_square, img_floor_corner, ground_img_size):
    pts_dst = np.array([[0,0],
               [0,1],
               [1,1],
               [1,0]
              ], dtype=float)

    h, status = cv2.findHomography(img_corner_square, pts_dst)
    
    poing_hom = np.ones((img_floor_corner.shape[0], 3))
    poing_hom[:, :2] = img_floor_corner
    poing_hom = poing_hom.T
    
    groundview_points = h @ poing_hom
    groundview_points = (groundview_points[:-1] / groundview_points[-1]).T
    
    #normalize between zero and one
    groundview_points = groundview_points - np.min(groundview_points, axis=0)
    groundview_points = groundview_points / np.max(groundview_points, axis=0)
    
    #rescale to fit in ground_img_size
    groundview_points = groundview_points * np.expand_dims(np.array(ground_img_size), axis=0)
    
    h, status = cv2.findHomography(img_floor_corner, groundview_points)
    
    h = np.linalg.inv(h)

    return h

def project_points(points, homography):

    poing_hom = np.ones((points.shape[0], points.shape[1] + 1))
    poing_hom[:, :2] = points
    poing_hom = poing_hom.T

    projected_points = homography @ poing_hom
    projected_points = (projected_points[:-1] / projected_points[-1]).T

    return  projected_points
    

def project_points_with_resize(points, homography, hom_input_size, hom_output_size, points_input_size, point_output_size):
    #Add rescaling to homography to account for reference frame size
    S1 = np.eye(3)
    S1[0,0] = point_output_size[1] / hom_output_size[1]
    S1[1,1] = point_output_size[0] / hom_output_size[0]

    S2 = np.eye(3)
    S2[0,0] = hom_input_size[1] / points_input_size[1] 
    S2[1,1] = hom_input_size[0] / points_input_size[0] 

    homography_with_resize = S1 @ homography @ S2
    
    return project_points(points, homography_with_resize)

    
def project_image_points_to_groundview(points, ground_plane_homography):

    H_inv = np.linalg.inv(ground_plane_homography)

    poing_hom = np.ones((points.shape[0], points.shape[1] + 1))
    poing_hom[:, :2] = points
    poing_hom = poing_hom.T

    groundview_points = H_inv.dot(poing_hom)
    groundview_points = (groundview_points[:-1] / groundview_points[-1]).T

    return  groundview_points


# def reproject_to_world_ground(ground_pix, K0, R0, T0):
#     """
#     Compute world coordinate from pixel coordinate of point on the groundplane
#     """
#     C0 = -R0.T @ T0
#     l = R0.T @ np.linalg.inv(K0) @ ground_pix
#     world_point = C0 - l*(C0[2]/l[2])
    
#     return world_point
    
def reproject_to_world_ground(ground_pix, K0, R0, T0, dist=None, height=0):
    """
    Compute world coordinate from pixel coordinate of point on a plane at specified height
    
    Args:
    ground_pix (array-like): The pixel coordinates of a point in the image
    K0 (array-like): The camera intrinsic matrix
    R0 (array-like): The camera rotation matrix
    T0 (array-like): The camera translation vector
    height (float): The height of the plane in world coordinates (default: 0)
    
    Returns:
    array-like: The 3D world coordinates of the point
    """

    if dist is not None:
        ground_pix = cv2.undistortPoints(np.array(ground_pix, dtype=np.float32), K0, dist, P=K0).reshape(2, 1)

    if ground_pix.shape[0] == 2:
        ground_pix = np.vstack((ground_pix, np.ones((1, ground_pix.shape[1]))))

        
    C0 = -R0.T @ T0
    l = R0.T @ np.linalg.inv(K0) @ ground_pix
    
    # Calculate the scaling factor to reach the specified height
    scale = (height - C0[2]) / l[2]
    
    world_point = C0 + l * scale
    
    return world_point


def reproject_to_world_ground_batched(ground_pix, K0, R0, T0, dist=None, height=0):
    """
    Compute world coordinates from pixel coordinates of points on a plane at specified height
    
    Args:
    ground_pix (array-like): The pixel coordinates of points in the image. Shape (N,2) or (N,3)
    K0 (array-like): The camera intrinsic matrix 
    R0 (array-like): The camera rotation matrix
    T0 (array-like): The camera translation vector
    height (float): The height of the plane in world coordinates (default: 0)
    
    Returns:
    array-like: The 3D world coordinates of the points. Shape (N,3)
    """


    if dist is not None:
        undistorted_points = cv2.undistortPoints(np.array(ground_pix, dtype=np.float32), K0, dist, P=K0)
        undistorted_points = undistorted_points.reshape(-1, 2)
        ground_pix = undistorted_points

    if ground_pix.shape[1] == 2:
        ground_pix_hom = np.hstack((ground_pix, np.ones((ground_pix.shape[0], 1))))
    else:
        ground_pix_hom = ground_pix
        
    # Transpose to 3xN for matrix operations
    ground_pix_hom = ground_pix_hom.T
        
    C0 = -R0.T @ T0
    
    K0_inv = np.linalg.inv(K0)
    
    l = R0.T @ K0_inv @ ground_pix_hom
    
    # Calculate the scaling factor to reach the specified height
    scale = (height - C0[2]) / l[2,:]
    
    # Reshape scale to broadcast correctly
    scale = scale[np.newaxis,:]
    
    # Broadcast C0 to match dimensions
    C0_expanded = np.repeat(C0[:,np.newaxis], ground_pix_hom.shape[1], axis=1)
    
    world_points = C0_expanded + l * scale
    
    # Transpose back to Nx3
    return world_points.T


def get_ray_directions(points_2d:np.ndarray, calib):
    points_2d = np.array(points_2d, dtype=float).reshape(-1, 1, 2)

    # Vectorized undistortion
    undistorted_points = cv2.undistortPoints(points_2d, calib.K, calib.dist, P=calib.K)
    undistorted_points = undistorted_points.reshape(-1, 2)

    # Homogeneous coordinates
    homogenous = np.hstack([undistorted_points, np.ones((len(undistorted_points), 1))])

    # Precompute matrices
    R_T = calib.R.T
    K_inv = np.linalg.inv(calib.K)
    ray_origin = (-R_T @ calib.T).flatten()

    # Compute ray directions
    temp = K_inv @ homogenous.T
    ray_directions = (R_T @ temp).T

    # Expand ray_origin to match number of points
    ray_origins = np.tile(ray_origin, (len(points_2d), 1))

    return ray_origins, ray_directions


def project_2d_points_to_mesh(points_2d, calib, mesh, VERBOSE=False, min_z=-4, max_z=1, min_cam_dist=1, z_plane=0.1):
    # Get ray origins and directions
    ray_origins, ray_directions = get_ray_directions(points_2d, calib)
    
    # Perform ray-mesh intersections
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=True
    )

    num_rays = len(ray_origins)
    ground_points = np.full((num_rays, 3), None)  # Initialize with None to allow filtering

    # Check if there are no intersections
    if len(locations) == 0:
        if VERBOSE:
            print(f"No intersections found for any rays. Projecting to groundplane. (z={z_plane})")
        ground_points = reproject_to_world_ground(points_2d, calib.K, calib.R, calib.T, calib.dist, z_plane)
        return ground_points#.tolist()

    # Cache variables to reduce attribute lookups
    R, T = calib.R, calib.T.reshape(3, 1)
    camera_coords = - (R @ locations.T) - T
    depths = np.abs(camera_coords[2, :])

    if VERBOSE:
        print(f"Depths sample: {depths[:5]}")

    # Use NumPy to efficiently process the intersections

    # Filter intersections by depth and z-coordinates in a single pass
    valid_mask = (depths > min_cam_dist) & (locations[:, 2] < max_z) & (locations[:, 2] > min_z)
    
    if VERBOSE: 
        print(f"ALL points: {locations}")
        print(f"Valid points: {locations[valid_mask]}")

    # Get valid indices per ray
    valid_indices = index_ray[valid_mask]
    valid_depths = depths[valid_mask]
    valid_points = locations[valid_mask]

    # Group and find the closest point for each ray
    if len(valid_indices) > 0:
        unique_rays, inverse_indices = np.unique(valid_indices, return_inverse=True)
        
        # For each unique ray, find the minimum depth and corresponding point
        min_depths_per_ray = np.full(num_rays, np.inf)
        closest_points = np.full((num_rays, 3), None)
        
        for i, ray_idx in enumerate(unique_rays):
            # Get depths and points for the current ray
            ray_depths = valid_depths[inverse_indices == i]
            ray_points = valid_points[inverse_indices == i]

            # Find the closest point based on minimum depth
            closest_idx = np.argmin(ray_depths)
            closest_points[ray_idx] = ray_points[closest_idx]
        
        # Assign only those rays that had valid intersections
        ground_points[unique_rays] = closest_points[unique_rays]
    else:
        if VERBOSE:
            print(f"No valid intersections after filtering. Projecting to groundplane. (z={z_plane})")
        ground_points = reproject_to_world_ground(points_2d, calib.K, calib.R, calib.T, calib.dist, z_plane)

    if VERBOSE:
        print(f"Ground points: {ground_points}")
    
    return ground_points#.tolist()
    

def project_world_to_camera(world_point, K1, R1, T1):
    """
    Project 3D point world coordinate to image plane (pixel coordinate)
    """
    point1 = ((R1 @ world_point) + T1)
    if(np.min(point1[2]) < 0 ):
        log.spam("Projection of world point located behind the camera plane")
    point1 = K1 @ point1
    point1 = point1 / point1[2]
    point1 = point1 / point1[2]
    
    return point1[:2]




def triangulate_point(points_2d, multi_calib):
    #Need at least point of view
    assert points_2d.shape[0] > 1
    
    #compute camera position for each view
    camera_positions = [-calib.R.T @ calib.T for calib in multi_calib]
    
    #Compute 3D direction from camera toward point
    point_directions = [-calib.R.T @ np.linalg.inv(calib.K) @ point for point, calib in zip(points_2d, multi_calib)]
    
    point_3d = nearest_intersection(np.array(camera_positions).squeeze(2), np.array(point_directions))
    
    return point_3d


def nearest_intersection(points, dirs):
    """
    :param points: (N, 3) array of points on the lines
    :param dirs: (N, 3) array of unit direction vectors
    :returns: (3,) array of intersection point
    
    from https://stackoverflow.com/questions/52088966/nearest-intersection-point-to-many-lines-in-python
    """
    #normalized direction
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_mat = dirs[:, :, np.newaxis] @ dirs[:, np.newaxis, :]
    points_mat = points[:, :, np.newaxis]
    I = np.eye(3)
    return np.linalg.lstsq(
        (I - dirs_mat).sum(axis=0),
        ((I - dirs_mat) @ points_mat).sum(axis=0),
        rcond=None
    )[0]


def project_roi_world_to_camera(world_point, K1, R1, T1):
    """
    Project Region of interest 3D point world coordinate to image plane (pixel coordinate)
    A bit Hacky since world coordinate are sometime behind image plane, we interpolate between corner of polygon
    to only keep point in front of the image plane
    """

    point1 = ((R1 @ world_point) + T1)

    if point1[2].min() < 0:
        #If a corner point of the roi lie behind the image compute corespondence in the image plane
        x = world_point[0]
        y = world_point[1]

        # Evenly sample point around polygon define by corner point in world_point
        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
        distance = distance/distance[-1]

        fx, fy = interp1d( distance, x ), interp1d( distance, y )

        alpha = np.linspace(0, 1, 150)
        x_regular, y_regular = fx(alpha), fy(alpha)

        world_point = np.vstack([x_regular, y_regular, np.zeros(x_regular.shape)])

        point1 = ((R1 @ world_point) + T1)
        
        #Filter out point behind the camera plane (Z < 0)    
        point1 = np.delete(point1, point1[2] < 0, axis=1)
    point1 = K1 @ point1
    point1 = point1 / point1[2]
    
    return point1[:2]


def update_img_point_boundary(img_points, view_ground_edge):
    #Make sure that all the img point are inside the image, if there are not replace them by points on the boundary
    img_points = map(Point, img_points)
    # img_corners = map(Point, [(0.0, 0.0), (0.0, img_size[0]), (img_size[1], img_size[0]), (img_size[1], 0.0)])
    img_corners = map(Point, view_ground_edge)

    poly1 = Polygon(*img_points)
    poly2 = Polygon(*img_corners)
    isIntersection = intersection(poly1, poly2)# poly1.intersection(poly2)
    
    point_inside = list(isIntersection)
    point_inside.extend([p for p in poly1.vertices if poly2.encloses_point(p)])
    point_inside.extend([p for p in poly2.vertices if poly1.encloses_point(p)])
   
    boundary_updated = convex_hull(*point_inside).vertices    
    boundary_updated = [p.coordinates for p in boundary_updated]

    return np.stack(boundary_updated).astype(float)


def project_world_point_to_groundplane(world_point, view_ground_edge, calib, H):
    """
    Project a point in 3d world coordinate lying on the ground (z=0), to it's 2D coordinate in the groundplane representation defined by world origin shift and rescale factor
    """
    
    #Project world coordinate to image plane (low res)
    point1 = ((calib.R @ world_point.T) + calib.T)
    point1 = calib.K @ point1
    # if(np.min(point1[2]) < 0 ):
    #     log.warning("Projecting world_point that are behind the camera to groundplane", np.min(point1[2]))
    # point1 = point1 / np.abs(point1[2])
    point1 = point1 / point1[2]

    # point1 = point1 / point1[2]
    point1 = point1[:2].T
    
    #compute intersection with ROI in groundplane, and original image in groundplane
    
    if H is not None:
        #project point from image coordinate to groundplane
        if len(view_ground_edge) != 0:
            view_ground_edge = project_image_points_to_groundview(view_ground_edge, H)
        point1 = project_image_points_to_groundview(point1, H)
    #     point_ground = H @ point1
    if len(view_ground_edge) != 0:
        point1 = update_img_point_boundary(point1, view_ground_edge)
    
    return point1




def generate_roi_and_boundary_from_corner_points(hm_size, org_size, corner_points, view_ground_edge, calib, H, is_world_coordinate=False, image_plane=False):
    """
    Generate regions of interest and region boundary from a set of corner point, if is_world_coordinate is true, 
    it assumes corner_points are 3d world cordinate and first project them to their 2d grounplane coordinate.
    """
    corner_points = np.array(corner_points)

    if is_world_coordinate:
        #convert world point to point in ground plane
        corner_points = project_world_point_to_groundplane(corner_points, view_ground_edge, calib, H)

    if image_plane:
        #if working in the image plane downscale roi point to be in the same scale as heatmaps
        corner_points = corner_points / 8

    poly = np.array(corner_points)
    # poly = np.around(poly)

    mask_boundary = np.zeros(hm_size)
    rr, cc = polygon_perimeter(poly[:,1], poly[:,0], mask_boundary.shape)
    mask_boundary[rr,cc] = 1
    mask_boundary = mask_boundary

    roi = np.zeros(hm_size)
    # we put mask boundary into ROI to make sure they overlap
    # roi[rr,cc] = 1
    rr, cc = polygon(poly[:,1], poly[:,0], roi.shape)
    roi[rr,cc] = 1

    return roi, mask_boundary


def remove_occluded_area_from_mask(mask, second_mask, occluded_area_list, view_ground_edge, org_size, calib, H, is_world_coordinate=True, image_plane=False):
    """
    mask: array to edit
    occluded_area_list: a list where each element corespond to an occluded area. each occluded area is representedn by its corner points.
    """

    for occluded_area in occluded_area_list:
        corner_points = np.array(occluded_area)

        if is_world_coordinate:
            #convert world point to point in ground plane
            corner_points = project_world_point_to_groundplane(corner_points, view_ground_edge, calib, H)

        if image_plane:
            #if working in the image plane downscale roi point to be in the same scale as heatmaps
            corner_points = corner_points / 8

        poly = np.array(corner_points)
        rr, cc = polygon(poly[:,1], poly[:,0], mask.shape)
        mask[rr,cc] = 0
        second_mask[rr,cc] = 0


    return mask, second_mask


def update_K_after_resize(K, old_size, new_size):
    fx = 1.0 / (old_size[1] / new_size[1])
    fy = 1.0 / (old_size[0] / new_size[0])

    scaler = np.array([
        [fx, 0, 0],
        [0, fy, 0],
        [0, 0, 1]]
    )

    new_K = scaler @ K

    return new_K

def rescale_keypoints(points, org_img_dim, out_img_dim):

    if len(points) == 0:
        return points
    
    out_img_dim = np.array(out_img_dim)
    org_img_dim = np.array(org_img_dim)
    
    if np.all(org_img_dim == out_img_dim):
        return points
    
    resize_factor = out_img_dim / org_img_dim
    #swap x and y
    resize_factor = resize_factor[::-1]

    resized_points = points*resize_factor

    return resized_points

def distance_point_to_line(lp1, lp2, p3):
    #Both point of the line are the same return distance to that point
    if np.all(lp1 == lp2):
        return np.linalg.norm(p3-lp1)
    else:
        return np.abs(np.cross(lp2-lp1, lp1-p3) / np.linalg.norm(lp2-lp1))
    

def load_mesh(mesh_path):
    """Load the scene mesh from the given path."""
    try:
        log.info(f"Loading mesh file from {mesh_path}")
        return trimesh.load(mesh_path)
    except Exception as e:
        log.error(f"Error loading mesh file: {e}")
        return None