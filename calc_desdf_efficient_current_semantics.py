#from utils.generate_desdf import raycast_desdf
from utils.utils import ray_cast


import matplotlib.image as mpimg
import numpy as np

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys

from joblib import Parallel, delayed
import multiprocessing

import time

def raycast_desdf(occ, sem, orn_slice=36, max_dist=10, original_resolution=0.01, resolution=0.1):
    ratio = resolution / original_resolution
    h, w = occ.shape
    desdf_shape = (int(h // ratio), int(w // ratio), orn_slice)
    desdf = np.zeros(desdf_shape)
    desdf_sem = np.zeros(desdf_shape)

    # Function to process each orientation
    def process_orientation(o):
        theta = o / orn_slice * 2 * np.pi
        local_desdf = np.zeros((desdf.shape[0], desdf.shape[1]))
        local_sem = np.zeros((desdf.shape[0], desdf.shape[1]))

        with tqdm(total=desdf.shape[0], desc=f'Orientation {o}/{orn_slice}', position=o+1, leave=False) as pbar:
            for row in range(0, desdf.shape[0]):
                for col in range(desdf.shape[1]):
                    pos = np.array([row, col]) * ratio
                    local_desdf[row, col], current_pos = ray_cast_extended(occ, pos, theta, max_dist / original_resolution)
                    current_pos_rounded = np.floor(current_pos).astype(int)
                    local_sem[row, col] = sem[current_pos_rounded[0], current_pos_rounded[1]]

                pbar.update(1)  # Update progress bar for each row processed

        return local_desdf, local_sem

    num_cores = multiprocessing.cpu_count()

    # Parallel processing with progress updates using tqdm
    results = Parallel(n_jobs=num_cores)(delayed(process_orientation)(o) for o in range(orn_slice))
    #results_desdf, results_sem = Parallel(n_jobs=num_cores)(delayed(process_orientation)(o) for o in range(orn_slice))
    
    # Unpack the results into desdf and desdf_sem
    for o, (result_desdf, result_sem) in enumerate(results):
        desdf[:, :, o] = result_desdf
        desdf_sem[:, :, o] = result_sem

    # for o, result_desdf in enumerate(results_desdf):
    #     desdf[:, :, o] = result_desdf
    # 
    # for o, result_sem in enumerate(results_sem):
    #     desdf_sem[:, :, o] = result_sem

    return desdf * original_resolution, desdf_sem





def ray_cast_extended(occ, pos, ang, dist_max=500):
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
                return dist_max, current_pos
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist, current_pos
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
        return dist, current_pos
    elif c == -1:
        # go left
        hit = False
        current_pos = pos.copy()
        while not hit:
            current_pos[1] -= 1
            if current_pos[1] < 0:
                return dist_max, current_pos
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist, current_pos
    elif s == -1:
        # go down
        hit = False
        current_pos = pos.copy()
        while not hit:
            current_pos[0] -= 1
            if current_pos[0] < 0:
                return dist_max, current_pos
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist, current_pos

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
                return dist_max, current_pos
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True

        dist = np.linalg.norm(current_pos - pos, 2)
        return dist, current_pos

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
                return dist_max, current_pos
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True

        dist = np.linalg.norm(current_pos - pos, 2)
        return dist, current_pos

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
                return dist_max, current_pos
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True

        dist = np.linalg.norm(current_pos - pos, 2)
        return dist, current_pos
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
                return dist_max, current_pos
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist, current_pos



if __name__ == "__main__":

    # Load and prepare floorplan with semantics
    floorplan_name = "HG_E_VF_20240215_preparing_occupancy_windows_semantics_rgb.png" #"map.png" #  "map_cropped.png" #
    floorplan = mpimg.imread(floorplan_name)

    ####
    floorplan_rounded = np.round(floorplan)

    # Extract semantics
    map_semantics = np.zeros(floorplan_rounded.shape[:2], dtype=int)
    map_semantics[np.all(floorplan_rounded == [0, 0, 0], axis=-1)] = 1  # [0, 0, 0] -> 1 (wall)
    map_semantics[np.all(floorplan_rounded == [1, 0, 0], axis=-1)] = 2  # [1, 0, 0] -> 2 (door)
    map_semantics[np.all(floorplan_rounded == [0, 1, 0], axis=-1)] = 3  # [0, 1, 0] -> 3 (elevator)

    # Extract occupancy map
    map_occupancy = np.ones(floorplan_rounded.shape[:2], dtype=int)
    map_occupancy[map_semantics==1] = 0
    map_occupancy[map_semantics==2] = 0
    map_occupancy[map_semantics==3] = 0

    ####

    pixel_per_meter = 18.315046895211292
    #occ = floorplan*255 # occ: the map as occupancy
    occ = map_occupancy*255
    original_resolution = 1/pixel_per_meter
    # resolution = 0.1
    # orn_slice = 72 # 36 # 144 #
    resolution = 0.25#0.5 #0.1
    orn_slice = 144 #72 #36 #
    max_dist = 1

    ### ORIGINAL
    desdf, desdf_sem = raycast_desdf(occ, map_semantics, orn_slice=orn_slice, max_dist=max_dist, original_resolution=original_resolution, resolution=resolution)
    filepath = "/cluster/project/cvg/data/lamar/HGE_customized_complete/hge_customized_complete/"
    filename = "desdf_complete_orn_slice_144_resolution_025.npy" #"desdf_complete_orn_slice_72_resolution_01.npy"
    np.save(filepath+filename, desdf)
    filename_sem = "desdf_sem_complete_orn_slice_144_resolution_025.npy" #"desdf_sem_complete_orn_slice_72_resolution_01.npy"
    np.save(filepath+filename_sem, desdf_sem)
    ###


