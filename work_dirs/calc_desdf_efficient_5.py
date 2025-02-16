from utils.generate_desdf import raycast_desdf
from utils.utils import ray_cast


import matplotlib.image as mpimg
import numpy as np

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm



import numpy as np
import tqdm

def raycast_desdf(
    occ, orn_slice=36, max_dist=10, original_resolution=0.01, resolution=0.1
):
    ratio = resolution / original_resolution
    desdf_shape = tuple((np.array(occ.shape) // ratio).astype(int)) + (orn_slice,)
    desdf = np.zeros(desdf_shape)

    # Iterate through orientations
    for o in tqdm.tqdm(range(orn_slice)):
        theta = o / orn_slice * np.pi * 2
        # Precompute trigonometric values
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Iterate through all pixels using numpy indexing
        for row in range(desdf.shape[0]):
            for col in range(desdf.shape[1]):
                pos = np.array([row, col]) * ratio
                desdf[row, col, o] = ray_cast(
                    occ, pos, cos_theta, sin_theta, theta, max_dist / original_resolution
                )

    return desdf * original_resolution



import numpy as np

def ray_cast(occ, pos, cos_theta, sin_theta, theta, dist_max=500):
    h, w = occ.shape

    # Flip occupancy map (assuming it's binary: 0 and 255)
    occ_flipped = 255 - occ

    # Determine direction of movement based on cos(theta) and sin(theta)
    if np.abs(cos_theta) > np.abs(sin_theta):
        step_x = np.sign(cos_theta)
        step_y = np.tan(theta) * step_x
    else:
        step_y = np.sign(sin_theta)
        step_x = step_y / np.tan(theta)

    # Initialize current position and distance
    current_pos = pos.copy()
    dist = 0

    while True:
        # Move to the next pixel
        current_pos[0] += step_y
        current_pos[1] += step_x

        # Check boundaries
        if (current_pos[0] < 0 or current_pos[0] >= h or
            current_pos[1] < 0 or current_pos[1] >= w):
            return dist_max

        # Convert to integer indices
        int_pos = np.floor(current_pos).astype(int)

        # Check if hit
        if occ_flipped[int_pos[0], int_pos[1]] > 0:
            return np.linalg.norm(current_pos - pos)

        # Increment distance
        dist += 1

        # Check maximum distance
        if dist > dist_max:
            return dist_max



if __name__ == "__main__":
    floorplan_name = "map_cropped.png" #"map.png" #  
    floorplan = mpimg.imread(floorplan_name)
    pixel_per_meter = 18.315046895211292
    occ = floorplan*255 # occ: the map as occupancy
    original_resolution = 1/pixel_per_meter
    resolution = 1
    orn_slice = 144
    max_dist = 10


    ### ORIGINAL
    desdf = raycast_desdf(occ, orn_slice=orn_slice, max_dist=max_dist, original_resolution=original_resolution, resolution=resolution)
    filename = "desdf_cropped_orn_slice_144_resolution_attempt_2.npy"
    np.save(filename, desdf)

    ###


