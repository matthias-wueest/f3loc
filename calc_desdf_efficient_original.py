from utils.generate_desdf import raycast_desdf
from utils.utils import ray_cast


import matplotlib.image as mpimg
import numpy as np

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm



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
    # assume occ resolution is 0.01
    ratio = resolution / original_resolution
    desdf = np.zeros(list((np.array(occ.shape) // ratio).astype(int)) + [orn_slice])
    # iterate through orientations
    for o in tqdm.tqdm(range(orn_slice)):
        theta = o / orn_slice * np.pi * 2
        # iterate through all pixels
        for row in tqdm.tqdm(range(desdf.shape[0])):
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
    # determine the first corner
    c = np.cos(ang)
    s = np.sin(ang)

    if c == 1:
        # go right
        hit = False
        current_pos = pos.copy()
        while not hit:
            current_pos[1] += 1
            if current_pos[1] >= w:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist
    elif s == 1:
        # go up
        hit = False
        current_pos = pos.copy()
        while not hit:
            current_pos[0] += 1
            if current_pos[0] >= h:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist
    elif c == -1:
        # go left
        hit = False
        current_pos = pos.copy()
        while not hit:
            current_pos[1] -= 1
            if current_pos[1] < 0:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist
    elif s == -1:
        # go down
        hit = False
        current_pos = pos.copy()
        while not hit:
            current_pos[0] -= 1
            if current_pos[0] < 0:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist

    if c > 0 and s > 0:
        corner = np.array([np.floor(pos[0] + 1), np.floor(pos[1] + 1)])
        # go up and right
        hit = False
        current_pos = pos.copy()
        while not hit:
            dw = corner[1] - current_pos[1]
            dh = corner[0] - current_pos[0]
            corner_ang = dh / dw
            if np.tan(ang) > corner_ang:
                # increment upwards
                current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                corner[0] += 1
            elif np.tan(ang) < corner_ang:
                # increment right
                current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                corner[1] += 1
            else:
                # increment both upwards and right
                current_pos = corner.copy()
                corner[0] += 1
                corner[1] += 1
            if current_pos[0] >= h or current_pos[1] >= w:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True

        dist = np.linalg.norm(current_pos - pos, 2)
        return dist

    elif c < 0 and s > 0:
        corner = np.array([np.floor(pos[0] + 1), np.ceil(pos[1] - 1)])
        # go up and left
        hit = False
        current_pos = pos.copy()
        while not hit:
            dw = corner[1] - current_pos[1]
            dh = corner[0] - current_pos[0]
            corner_ang = dh / dw
            if np.tan(ang) < corner_ang:
                # increment upwards
                current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                corner[0] += 1
            elif np.tan(ang) > corner_ang:
                # increment left
                current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                corner[1] -= 1
            else:
                # increment both upwards and left
                current_pos = corner.copy()
                corner[0] += 1
                corner[1] -= 1
            if current_pos[0] >= h or current_pos[1] < 0:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True

        dist = np.linalg.norm(current_pos - pos, 2)
        return dist

    elif c < 0 and s < 0:
        corner = np.array([np.ceil(pos[0] - 1), np.ceil(pos[1] - 1)])
        # go down and left
        hit = False
        current_pos = pos.copy()
        while not hit:
            dw = corner[1] - current_pos[1]
            dh = corner[0] - current_pos[0]
            corner_ang = dh / dw
            if np.tan(ang) > corner_ang:
                # increment downwards
                current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                corner[0] -= 1
            elif np.tan(ang) < corner_ang:
                # increment left
                current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                corner[1] -= 1
            else:
                # increment both downwards and left
                current_pos = corner.copy()
                corner[0] -= 1
                corner[1] -= 1
            if current_pos[0] < 0 or current_pos[1] < 0:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True

        dist = np.linalg.norm(current_pos - pos, 2)
        return dist
    elif c > 0 and s < 0:
        corner = np.array([np.ceil(pos[0] - 1), np.floor(pos[1] + 1)])
        # go down and right
        hit = False
        current_pos = pos.copy()
        while not hit:
            dw = corner[1] - current_pos[1]
            dh = corner[0] - current_pos[0]
            corner_ang = dh / dw
            if np.tan(ang) < corner_ang:
                # increment downwards
                current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                corner[0] -= 1
            elif np.tan(ang) > corner_ang:
                # increment right
                current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                corner[1] += 1
            else:
                # increment both downwards and right
                current_pos = corner.copy()
                corner[0] -= 1
                corner[1] += 1
            if current_pos[0] < 0 or current_pos[1] >= w:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist



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


