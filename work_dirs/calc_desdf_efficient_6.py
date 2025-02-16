import matplotlib.image as mpimg
import numpy as np
import tqdm

from utils.generate_desdf import raycast_desdf  # Assuming this is for some other functionality
from utils.utils import ray_cast


def raycast_desdf(
    occ, orn_slice=36, max_dist=10, original_resolution=0.01, resolution=0.1
):
    ratio = resolution / original_resolution
    desdf_shape = tuple((np.array(occ.shape) // ratio).astype(int)) + (orn_slice,)
    desdf = np.zeros(desdf_shape)

    BATCH_SIZE = 100  # Adjust batch size based on your memory constraints

    for o in tqdm.trange(orn_slice, desc='Orientation', unit='slice'):
        theta = o / orn_slice * np.pi * 2
        for row in tqdm.trange(desdf.shape[0], desc='Row', unit='row', leave=False):
            for col in range(0, desdf.shape[1], BATCH_SIZE):  # Process in batches
                col_end = min(col + BATCH_SIZE, desdf.shape[1])
                pos_batch = (np.indices((row, col_end - col)) * ratio).reshape(2, -1).T + np.array([row, col]) * ratio
                desdf[row, col:col_end, o] = ray_cast_batch(
                    occ, pos_batch, theta, max_dist / original_resolution
                )

    return desdf * original_resolution


def ray_cast_batch(occ, pos_batch, ang, dist_max=500):
    dist_batch = np.full(len(pos_batch), dist_max)
    h, w = occ.shape
    occ_flipped = 255 - occ

    for i, pos in enumerate(pos_batch):
        current_pos = pos.copy()

        c = np.cos(ang)
        s = np.sin(ang)

        if c == 1:
            while current_pos[1] < w:
                current_pos[1] += 1
                if current_pos[1] >= w or occ_flipped[int(current_pos[0]), int(current_pos[1])] > 0:
                    dist_batch[i] = np.linalg.norm(current_pos - pos, 2)
                    break

        elif s == 1:
            while current_pos[0] < h:
                current_pos[0] += 1
                if current_pos[0] >= h or occ_flipped[int(current_pos[0]), int(current_pos[1])] > 0:
                    dist_batch[i] = np.linalg.norm(current_pos - pos, 2)
                    break

        elif c == -1:
            while current_pos[1] >= 0:
                current_pos[1] -= 1
                if current_pos[1] < 0 or occ_flipped[int(current_pos[0]), int(current_pos[1])] > 0:
                    dist_batch[i] = np.linalg.norm(current_pos - pos, 2)
                    break

        elif s == -1:
            while current_pos[0] >= 0:
                current_pos[0] -= 1
                if current_pos[0] < 0 or occ_flipped[int(current_pos[0]), int(current_pos[1])] > 0:
                    dist_batch[i] = np.linalg.norm(current_pos - pos, 2)
                    break

        elif c > 0 and s > 0:
            corner = np.array([np.floor(pos[0] + 1), np.floor(pos[1] + 1)])
            while True:
                dw = corner[1] - current_pos[1]
                dh = corner[0] - current_pos[0]
                corner_ang = dh / dw
                if np.tan(ang) > corner_ang:
                    current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                    corner[0] += 1
                elif np.tan(ang) < corner_ang:
                    current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                    corner[1] += 1
                else:
                    current_pos = corner.copy()
                    corner[0] += 1
                    corner[1] += 1
                if current_pos[0] >= h or current_pos[1] >= w or occ_flipped[int(current_pos[0]), int(current_pos[1])] > 0:
                    dist_batch[i] = np.linalg.norm(current_pos - pos, 2)
                    break

        elif c < 0 and s > 0:
            corner = np.array([np.floor(pos[0] + 1), np.ceil(pos[1] - 1)])
            while True:
                dw = corner[1] - current_pos[1]
                dh = corner[0] - current_pos[0]
                corner_ang = dh / dw
                if np.tan(ang) < corner_ang:
                    current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                    corner[0] += 1
                elif np.tan(ang) > corner_ang:
                    current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                    corner[1] -= 1
                else:
                    current_pos = corner.copy()
                    corner[0] += 1
                    corner[1] -= 1
                if current_pos[0] >= h or current_pos[1] < 0 or occ_flipped[int(current_pos[0]), int(current_pos[1])] > 0:
                    dist_batch[i] = np.linalg.norm(current_pos - pos, 2)
                    break

        elif c < 0 and s < 0:
            corner = np.array([np.ceil(pos[0] - 1), np.ceil(pos[1] - 1)])
            while True:
                dw = corner[1] - current_pos[1]
                dh = corner[0] - current_pos[0]
                corner_ang = dh / dw
                if np.tan(ang) > corner_ang:
                    current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                    corner[0] -= 1
                elif np.tan(ang) < corner_ang:
                    current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                    corner[1] -= 1
                else:
                    current_pos = corner.copy()
                    corner[0] -= 1
                    corner[1] -= 1
                if current_pos[0] < 0 or current_pos[1] < 0 or occ_flipped[int(current_pos[0]), int(current_pos[1])] > 0:
                    dist_batch[i] = np.linalg.norm(current_pos - pos, 2)
                    break

        elif c > 0 and s < 0:
            corner = np.array([np.ceil(pos[0] - 1), np.floor(pos[1] + 1)])
            while True:
                dw = corner[1] - current_pos[1]
                dh = corner[0] - current_pos[0]
                corner_ang = dh / dw
                if np.tan(ang) < corner_ang:
                    current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                    corner[0] -= 1
                elif np.tan(ang) > corner_ang:
                    current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                    corner[1] += 1
                else:
                    current_pos = corner.copy()
                    corner[0] -= 1
                    corner[1] += 1
                if current_pos[0] < 0 or current_pos[1] >= w or occ_flipped[int(current_pos[0]), int(current_pos[1])] > 0:
                    dist_batch[i] = np.linalg.norm(current_pos - pos, 2)
                    break

    return dist_batch


if __name__ == "__main__":
    floorplan_name = "map_cropped.png"
    floorplan = mpimg.imread(floorplan_name)
    pixel_per_meter = 18.315046895211292
    occ = floorplan * 255
    original_resolution = 1 / pixel_per_meter
    resolution = 1
    orn_slice = 144
    max_dist = 10

    desdf = raycast_desdf(
        occ,
        orn_slice=orn_slice,
        max_dist=max_dist,
        original_resolution=original_resolution,
        resolution=resolution,
    )

    filename = "desdf_cropped_orn_slice_144_resolution_attempt_2.npy"
    np.save(filename, desdf)
