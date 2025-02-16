from numba import njit, prange
import numpy as np
import matplotlib.image as mpimg
import tqdm
import multiprocessing

@njit
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
    c = np.cos(ang)
    s = np.sin(ang)

    current_pos = pos.copy()
    while True:
        current_pos[0] += s
        current_pos[1] += c

        if current_pos[0] < 0 or current_pos[0] >= h or current_pos[1] < 0 or current_pos[1] >= w:
            return dist_max
        if occ[int(current_pos[0]), int(current_pos[1])] > 0:
            break

    dist = np.linalg.norm(current_pos - pos, 2)
    return dist


@njit(parallel=True)
def compute_desdf(occ, orn_slice, max_dist, original_resolution, resolution):
    ratio = resolution / original_resolution
    desdf = np.zeros((occ.shape[0] // int(ratio), occ.shape[1] // int(ratio), orn_slice))
    for o in prange(orn_slice):
        theta = o / orn_slice * np.pi * 2
        for row in prange(desdf.shape[0]):
            for col in prange(desdf.shape[1]):
                pos = np.array([row, col]) * ratio
                desdf[row, col, o] = ray_cast(occ, pos, theta, max_dist / original_resolution)
    return desdf * original_resolution


def raycast_desdf_with_progress(occ, orn_slice, max_dist, original_resolution, resolution):
    ratio = resolution / original_resolution
    desdf_shape = (occ.shape[0] // int(ratio), occ.shape[1] // int(ratio), orn_slice)
    desdf = np.zeros(desdf_shape)

    total_steps = orn_slice * desdf_shape[0] * desdf_shape[1]
    
    with tqdm.tqdm(total=total_steps) as pbar:
        for o in range(orn_slice):
            theta = o / orn_slice * np.pi * 2
            for row in range(desdf_shape[0]):
                for col in range(desdf_shape[1]):
                    pos = np.array([row, col]) * ratio
                    desdf[row, col, o] = ray_cast(occ, pos, theta, max_dist / original_resolution)
                    pbar.update(1)
                    
    return desdf * original_resolution


if __name__ == "__main__":
    floorplan_name = "map_cropped.png"  # "map.png" #
    floorplan = mpimg.imread(floorplan_name)
    pixel_per_meter = 18.315046895211292
    occ = floorplan * 255
    original_resolution = 1 / pixel_per_meter
    resolution = 1
    orn_slice = 144
    max_dist = 10

    desdf = raycast_desdf_with_progress(occ, orn_slice, max_dist, original_resolution, resolution)
    filename = "desdf_cropped_orn_slice_144_resolution_1_attempt.npy"  # "desdf_complete_orn_slice_36_resolution_01.npy" #
    np.save(filename, desdf)
