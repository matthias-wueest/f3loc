import matplotlib.image as mpimg
import numpy as np
import tqdm

# Assuming these are your existing imports
# from utils.generate_desdf import raycast_desdf
# from utils.utils import ray_cast
# import cv2
# import matplotlib.pyplot as plt

def raycast_desdf(
    occ, orn_slice=36, max_dist=10, original_resolution=0.01, resolution=0.1
):
    """
    Get desdf from occupancy grid through brute force raycast
    Input:
        occ: the map as occupancy
        orn_slice: number of equiangular orientations
        max_dist: maximum raycast distance, [m]
        original_resolution: the resolution of occ input [m/pixel]
        resolution: output resolution of the desdf [m/pixel]
    Output:
        desdf: the directional esdf of the occ input in meter
    """
    ratio = resolution / original_resolution
    desdf = np.zeros(list((np.array(occ.shape) // ratio).astype(int)) + [orn_slice])
    
    for o in tqdm.tqdm(range(orn_slice), desc='Orientation', position=0):
        theta = o / orn_slice * np.pi * 2
        
        for row in tqdm.tqdm(range(desdf.shape[0]), desc='Row', position=1, leave=False):
            for col in range(desdf.shape[1]):
                pos = np.array([row, col]) * ratio
                desdf[row, col, o] = ray_cast(
                    occ, pos, theta, max_dist / original_resolution
                )

    return desdf * original_resolution


def ray_cast(occ, pos, ang, dist_max=500):
    """
    Cast ray in the occupancy map
    Input:
        pos: in image coordinate, in pixel, [h, w]
        ang: ray shooting angle, in radian
    Output:
        dist: in pixels
    """
    h = occ.shape[0]
    w = occ.shape[1]
    occ = 255 - occ
    
    # Calculate cosine and sine of the angle
    c = np.cos(ang)
    s = np.sin(ang)
    
    # Initialize current position
    current_pos = pos.copy()
    
    while True:
        # Calculate the next position based on angle
        if np.abs(c) > np.abs(s):
            step = np.sign(c)
            if step > 0:
                next_pos = np.array([current_pos[0], np.ceil(current_pos[1])])
            else:
                next_pos = np.array([current_pos[0], np.floor(current_pos[1])])
            
            delta = (next_pos[1] - current_pos[1]) / c
        else:
            step = np.sign(s)
            if step > 0:
                next_pos = np.array([np.ceil(current_pos[0]), current_pos[1]])
            else:
                next_pos = np.array([np.floor(current_pos[0]), current_pos[1]])
            
            delta = (next_pos[0] - current_pos[0]) / s
        
        # Check if the next position is out of bounds
        if delta >= dist_max:
            return dist_max
        
        # Update the current position
        current_pos += delta * np.array([c, s])
        
        # Check for collision
        if (current_pos < 0).any() or (current_pos >= np.array([h, w])).any():
            return dist_max
        if occ[int(current_pos[0]), int(current_pos[1])] > 0:
            return np.linalg.norm(current_pos - pos)
        

if __name__ == "__main__":
    # Load the floorplan image
    floorplan_name = "map_cropped.png"
    floorplan = mpimg.imread(floorplan_name)
    
    # Convert floorplan to occupancy grid
    occ = floorplan * 255
    
    # Parameters for raycasting
    pixel_per_meter = 18.315046895211292
    original_resolution = 1 / pixel_per_meter
    resolution = 1
    orn_slice = 144
    max_dist = 10
    
    # Perform raycasting to generate desdf
    desdf = raycast_desdf(occ, orn_slice=orn_slice, max_dist=max_dist, original_resolution=original_resolution, resolution=resolution)
    
    # Save desdf to a file
    filename = "desdf_cropped_orn_slice_144_resolution_attempt_2.npy"
    np.save(filename, desdf)
