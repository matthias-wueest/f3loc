import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tqdm
from numba import jit, prange

@jit(nopython=True)
def ray_cast_optimized(occ, pos, ang, dist_max=500):
    h, w = occ.shape
    occ = 255 - occ
    c = np.cos(ang)
    s = np.sin(ang)
    dist = dist_max

    x, y = pos
    if c == 1:
        for j in range(int(pos[1]), w):
            if occ[int(x), j] > 0:
                return np.hypot(j - pos[1], 0)
    elif s == 1:
        for i in range(int(pos[0]), h):
            if occ[i, int(y)] > 0:
                return np.hypot(0, i - pos[0])
    elif c == -1:
        for j in range(int(pos[1]), -1, -1):
            if occ[int(x), j] > 0:
                return np.hypot(j - pos[1], 0)
    elif s == -1:
        for i in range(int(pos[0]), -1, -1):
            if occ[i, int(y)] > 0:
                return np.hypot(0, i - pos[0])
    else:
        while 0 <= x < h and 0 <= y < w:
            if occ[int(x), int(y)] > 0:
                dist = np.hypot(x - pos[0], y - pos[1])
                break
            x += c
            y += s

    return dist

@jit(nopython=True, parallel=True)
def raycast_slice(occ, ratio, theta, max_dist, original_resolution, start_row, end_row):
    slice_shape = ((end_row - start_row), occ.shape[1])
    slice_desdf = np.zeros((slice_shape[0], slice_shape[1]), dtype=np.float32)
    
    for row in prange(slice_shape[0]):
        for col in range(slice_shape[1]):
            pos = np.array([row + start_row, col]) * ratio
            slice_desdf[row, col] = ray_cast_optimized(occ, pos, theta, max_dist / original_resolution)
    
    return slice_desdf

def raycast_desdf_optimized(occ, orn_slice=36, max_dist=10, original_resolution=0.01, resolution=0.1):
    ratio = resolution / original_resolution
    new_shape = (np.array(occ.shape) // ratio).astype(np.int32)
    desdf = np.zeros((new_shape[0], new_shape[1], orn_slice), dtype=np.float32)
    
    for o in tqdm.tqdm(range(orn_slice), desc='Orientations'):
        theta = o / orn_slice * np.pi * 2
        for start_row in tqdm.tqdm(range(0, new_shape[0], 100), desc=f'Rows (Orientation {o+1}/{orn_slice})', leave=False):
            end_row = min(start_row + 100, new_shape[0])
            slice_desdf = raycast_slice(occ, ratio, theta, max_dist, original_resolution, start_row, end_row)
            print(f"Shape of slice_desdf: {slice_desdf.shape}, expected shape: {(end_row - start_row, new_shape[1])}")
            desdf[start_row:end_row, :, o] = slice_desdf

    return desdf * original_resolution

if __name__ == "__main__":
    floorplan_name = "map_cropped.png"
    floorplan = mpimg.imread(floorplan_name)
    pixel_per_meter = 18.315046895211292
    occ = (floorplan * 255).astype(np.uint8)
    original_resolution = 1 / pixel_per_meter
    resolution = 1
    orn_slice = 144
    max_dist = 10

    desdf = raycast_desdf_optimized(occ, orn_slice=orn_slice, max_dist=max_dist, original_resolution=original_resolution, resolution=resolution)
    filename = "desdf_cropped_orn_slice_144_resolution_1_attempt3.npy"
    np.save(filename, desdf)
