from numba import njit, prange
import numpy as np
import matplotlib.image as mpimg
import tqdm


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


#@njit(parallel=True)
def raycast_desdf_with_progress(occ, orn_slice=36, max_dist=10, original_resolution=0.01, resolution=0.1):
    ratio = resolution / original_resolution
    desdf = np.zeros(list((np.array(occ.shape) // ratio).astype(int)) + [orn_slice])
    with tqdm.tqdm(total=orn_slice * desdf.shape[0] * desdf.shape[1]) as pbar:
        for o in range(orn_slice):
            theta = o / orn_slice * np.pi * 2
            for row in range(desdf.shape[0]):
                for col in range(desdf.shape[1]):
                    pos = np.array([row, col]) * ratio
                    desdf[row, col, o] = ray_cast(occ, pos, theta, max_dist / original_resolution)
                    pbar.update(1)
    return desdf * original_resolution


if __name__ == "__main__":
    floorplan_name = "map_cropped.png" #"map.png" # 
    floorplan = mpimg.imread(floorplan_name)
    pixel_per_meter = 18.315046895211292
    occ = floorplan * 255
    original_resolution = 1 / pixel_per_meter
    resolution = 1
    orn_slice = 36
    max_dist = 10

    desdf = raycast_desdf_with_progress(occ, orn_slice=orn_slice, max_dist=max_dist, original_resolution=original_resolution, resolution=resolution)
    filename = "desdf_cropped_orn_slice_36_resolution_1_attempt2.npy" #"desdf_complete_orn_slice_36_resolution_01.npy" #
    np.save(filename, desdf)
