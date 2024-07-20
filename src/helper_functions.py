import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull



def find_affine_transform(pts_src, pts_dst):
    """
    Find the affine transformation that maps source points to destination points.
    
    Parameters:
    pts_src (ndarray): Source points of shape (4, 2).
    pts_dst (ndarray): Destination points of shape (4, 2).
    
    Returns:
    ndarray: Affine transformation matrix of shape (2, 3).
    """
    A = []
    B = []
    for (x, y), (x_prime, y_prime) in zip(pts_src, pts_dst):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.append(x_prime)
        B.append(y_prime)
    
    A = np.array(A)
    B = np.array(B)
    
    # Solve the linear system A * [a, b, c, d, e, f].T = B
    affine_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    # Reshape the result into a 2x3 matrix
    affine_matrix = affine_params.reshape(2, 3)
    
    return affine_matrix

# Apply the affine transformation to a point
def apply_affine_transformation(point, affine_matrix):
    point_augmented = np.append(point, 1)  # Convert to homogeneous coordinates
    transformed_point = np.dot(affine_matrix, point_augmented)
    return transformed_point

# Function to apply the rotation to an angle
def apply_rotation(angle, rotation_matrix):
    # Create a vector from the angle (assuming 2D)
    vec = np.array([np.cos(angle), np.sin(angle)])
    # Apply the rotation matrix to this vector
    transformed_vec = np.dot(rotation_matrix, vec)
    # Calculate the new angle from the transformed vector
    new_angle = np.arctan2(transformed_vec[1], transformed_vec[0])
    return new_angle


# Inverse affine transformation
def apply_inverse_affine_transformation(point, affine_matrix):
    # Calculate the inverse affine matrix
    affine_matrix_inverse = np.linalg.inv(np.vstack([affine_matrix, [0, 0, 1]]))[:2, :]
    point_augmented = np.append(point, 1)  # Convert to homogeneous coordinates
    transformed_point = np.dot(affine_matrix_inverse, point_augmented)
    return transformed_point


# Inverse rotation transformation
def apply_inverse_rotation(angle, rotation_matrix):
    # Transpose of the rotation matrix is the inverse for pure rotation matrices
    rotation_matrix_inverse = rotation_matrix.T
    # Create a vector from the angle (assuming 2D)
    vec = np.array([np.cos(angle), np.sin(angle)])
    # Apply the inverse rotation matrix to this vector
    transformed_vec = np.dot(rotation_matrix_inverse, vec)
    # Calculate the new angle from the transformed vector
    new_angle = np.arctan2(transformed_vec[1], transformed_vec[0])
    return new_angle


# Function to transform world coordinates to map coordinates
def world_to_map(position_world, orientation_world_rad, affine_matrix, floorplan_correspondences):
    # Apply the affine transformation
    position_map = apply_affine_transformation(position_world, affine_matrix)
    # Adjust the map position due to cropping
    position_map = position_map - floorplan_correspondences[0, :] + np.array([685, 20])
    
    # Adjust the orientation
    orientation_map_rad0 = np.pi / 2
    orientation_map_rad = apply_rotation(orientation_world_rad, affine_matrix[:, :2])
    orientation_map_rad = orientation_map_rad - orientation_map_rad0
    
    return position_map, orientation_map_rad


# Function to transform world coordinates to map coordinates
def world_to_map_hge_complete(position_world, orientation_world_rad, affine_matrix):
    # Apply the affine transformation
    position_map = apply_affine_transformation(position_world, affine_matrix)
    
    # Adjust the orientation
    orientation_map_rad0 = np.pi / 2
    orientation_map_rad = apply_rotation(orientation_world_rad, affine_matrix[:, :2])
    orientation_map_rad = orientation_map_rad - orientation_map_rad0
    
    return position_map, orientation_map_rad


# Function to transform map coordinates to world coordinates
def map_to_world(position_map, orientation_map_rad, affine_matrix, floorplan_correspondences):
    # Adjust the map position due to cropping
    position_map_adjusted = position_map + floorplan_correspondences[0, :] - np.array([685, 20])
    # Apply the inverse affine transformation
    position_world = apply_inverse_affine_transformation(position_map_adjusted, affine_matrix)

    # Adjust the orientation
    orientation_map_rad_adjusted = orientation_map_rad + np.pi / 2
    # Apply the inverse rotation transformation
    orientation_world_rad = apply_inverse_rotation(orientation_map_rad_adjusted, affine_matrix[:, :2])
    
    return position_world, orientation_world_rad


# Function to transform map coordinates to world coordinates
def map_to_world_hge_complete(position_map, orientation_map_rad, affine_matrix):
    # Apply the inverse affine transformation
    position_world = apply_inverse_affine_transformation(position_map, affine_matrix)

    # Adjust the orientation
    orientation_map_rad_adjusted = orientation_map_rad + np.pi / 2
    # Apply the inverse rotation transformation
    orientation_world_rad = apply_inverse_rotation(orientation_map_rad_adjusted, affine_matrix[:, :2])
    
    return position_world, orientation_world_rad



def minimum_bounding_box(points):

    if len(points) < 3:
        if len(points) == 2:
            width = np.linalg.norm(points[1] - points[0])
            return width  # Distance between two points
        else:
            return 0  # Single point, diagonal length is 0

    # Compute the convex hull
    hull = ConvexHull(points, qhull_options='QJ')
    hull_points = points[hull.vertices]
    
    # Initialize variables
    min_area = float('inf')
    best_box = None
    
    # Rotating calipers to find the minimum bounding box
    for i in range(len(hull_points)):
        # Calculate edge angle
        edge = hull_points[i] - hull_points[i - 1]
        edge_angle = np.arctan2(edge[1], edge[0])
        cos_angle = np.cos(edge_angle)
        sin_angle = np.sin(edge_angle)
        
        # Rotate the points
        rotation_matrix = np.array([
            [cos_angle, sin_angle],
            [-sin_angle, cos_angle]
        ])
        rot_points = np.dot(hull_points, rotation_matrix)
        
        # Find the bounding box
        min_x, min_y = np.min(rot_points, axis=0)
        max_x, max_y = np.max(rot_points, axis=0)
        area = (max_x - min_x) * (max_y - min_y)
        
        # Check if this is the minimum area
        if area < min_area:
            min_area = area
            best_box = (min_x, max_x, min_y, max_y, rotation_matrix)
    
    # # Extract the best box parameters
    # min_x, max_x, min_y, max_y, rotation_matrix = best_box
    # corner_points = np.array([
    #     [min_x, min_y],
    #     [min_x, max_y],
    #     [max_x, max_y],
    #     [max_x, min_y]
    # ])
    # corner_points = np.dot(corner_points, rotation_matrix.T)
    #
    # return corner_points, min_area
    
    width = max_x - min_x
    height = max_y - min_y
    diagonal_length = np.sqrt(width**2 + height**2)
    
    return diagonal_length

    