import cv2
import numpy as np


def gravity_align(
    img,
    r,
    p,
    K=np.array([[240, 0, 320], [0, 240, 240], [0, 0, 1]]).astype(np.float32),
    mode=0,
):
    """
    Align the image with gravity direction
    Input:
        img: input image
        r: roll
        p: pitch
        K: camera intrisics
        mode: interpolation mode for warping, default: 0 - 'linear', else 1 - 'nearest'
    Output:
        aligned_img: gravity aligned image
    """
    # calculate R_gc from roll and pitch
    # From gravity to camera, yaw->pitch->roll
    # From camera to gravity, roll->pitch->yaw
    p = (
        -p
    )  # this is because the pitch axis of robot and camera is in the opposite direction
    cr = np.cos(r)
    sr = np.sin(r)
    cp = np.cos(p)
    sp = np.sin(p)

    # compute R_cg first
    # pitch
    R_x = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])

    # roll
    R_z = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])

    R_cg = R_z @ R_x
    R_gc = R_cg.T

    # get shape
    h, w = list(img.shape[:2])

    # directly compute the homography
    persp_M = K @ R_gc @ np.linalg.inv(K)

    aligned_img = cv2.warpPerspective(
        img, persp_M, (w, h), flags=cv2.INTER_NEAREST if mode == 1 else cv2.INTER_LINEAR
    )

    return aligned_img


def gravity_align_depth(depth_img, r, p, K=np.array([[240, 0, 320], [0, 240, 240], [0, 0, 1]]).astype(np.float32)):
    """
    Align the depth image with gravity direction
    Input:
        depth_img: input depth image
        r: roll
        p: pitch
        K: camera intrinsics
    Output:
        aligned_depth_img: gravity aligned depth image
    """
    # Calculate R_gc from roll and pitch
    p = -p  # This is because the pitch axis of robot and camera is in the opposite direction
    cr = np.cos(r)
    sr = np.sin(r)
    cp = np.cos(p)
    sp = np.sin(p)

    # Compute R_cg first
    # Pitch
    R_x = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])

    # Roll
    R_z = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])

    R_cg = R_z @ R_x
    R_gc = R_cg.T

    # Get the shape of the depth image
    h, w = depth_img.shape

    # Generate grid of (u, v) pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Back-project to 3D
    z = depth_img.flatten()
    x = (u.flatten() - K[0, 2]) * z / K[0, 0]
    y = (v.flatten() - K[1, 2]) * z / K[1, 1]

    # Stack to get 3D points
    points_3D = np.vstack((x, y, z))

    # Rotate points
    rotated_points_3D = R_gc @ points_3D

    # Project back to 2D
    x_rot = (rotated_points_3D[0, :] * K[0, 0] / rotated_points_3D[2, :]) + K[0, 2]
    y_rot = (rotated_points_3D[1, :] * K[1, 1] / rotated_points_3D[2, :]) + K[1, 2]
    z_rot = rotated_points_3D[2, :]

    # Round and cast to int
    x_rot = np.round(x_rot).astype(int)
    y_rot = np.round(y_rot).astype(int)

    # Create an empty aligned depth image
    aligned_depth_img = np.zeros_like(depth_img)

    # Mask for valid indices
    valid_mask = (x_rot >= 0) & (x_rot < w) & (y_rot >= 0) & (y_rot < h)

    # Filter valid points
    x_rot = x_rot[valid_mask]
    y_rot = y_rot[valid_mask]
    z_rot = z_rot[valid_mask]

    # Use numpy's advanced indexing to assign the values
    aligned_depth_img[y_rot, x_rot] = z_rot

    # Create a mask with 1 for non-zero pixels and 0 for zero pixels
    mask = (aligned_depth_img > 0).astype(np.uint8)

    return aligned_depth_img, mask


def gravity_align_normal(normals_img, depth_img, r, p, K=np.array([[240, 0, 320], [0, 240, 240], [0, 0, 1]]).astype(np.float32)):
    """
    Align the normals image with gravity direction, both normal vectors and pixel positions
    Input:
        normals_img: input normals image (HxWx3)
        depth_img: input depth image (HxW)
        r: roll
        p: pitch
        K: camera intrinsics
    Output:
        aligned_normals_img: gravity-aligned normals image (HxWx3)
    """
    # Calculate R_gc from roll and pitch
    p = -p  # Adjust pitch as before
    cr = np.cos(r)
    sr = np.sin(r)
    cp = np.cos(p)
    sp = np.sin(p)

    # Compute R_cg first
    # Pitch
    R_x = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])

    # Roll
    R_z = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])

    R_cg = R_z @ R_x
    R_gc = R_cg.T  # Transpose to get R_gc

    # Get the shape of the normal and depth images
    h, w, _ = normals_img.shape

    # Generate grid of (u, v) pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Back-project to 3D using depth image
    z = depth_img.flatten()
    x = (u.flatten() - K[0, 2]) * z / K[0, 0]
    y = (v.flatten() - K[1, 2]) * z / K[1, 1]

    # Stack to get 3D points
    points_3D = np.vstack((x, y, z))

    # Rotate points
    rotated_points_3D = R_gc @ points_3D

    # Rotate normals
    normals_reshaped = normals_img.reshape(-1, 3).T
    rotated_normals = R_gc @ normals_reshaped

    # Project back to 2D
    x_rot = (rotated_points_3D[0, :] * K[0, 0] / rotated_points_3D[2, :]) + K[0, 2]
    y_rot = (rotated_points_3D[1, :] * K[1, 1] / rotated_points_3D[2, :]) + K[1, 2]

    # Round and cast to int
    x_rot = np.round(x_rot).astype(int)
    y_rot = np.round(y_rot).astype(int)

    # Create an empty aligned normal image
    aligned_normals_img = np.zeros_like(normals_img)

    # Mask for valid indices
    valid_mask = (x_rot >= 0) & (x_rot < w) & (y_rot >= 0) & (y_rot < h)

    # Filter valid points
    x_rot = x_rot[valid_mask]
    y_rot = y_rot[valid_mask]
    rotated_normals = rotated_normals[:, valid_mask]

    # Use numpy's advanced indexing to assign the rotated normals
    aligned_normals_img[y_rot, x_rot, :] = rotated_normals.T

    return aligned_normals_img


def gravity_align_segmentation(
    seg_map,
    r,
    p,
    K=np.array([[240, 0, 320], [0, 240, 240], [0, 0, 1]]).astype(np.float32),
    mode=0
):
    """
    Align the segmentation map with gravity direction.
    
    Input:
        seg_map: input segmentation map of shape (N, H, W), where N is the number of channels.
        r: roll angle in radians.
        p: pitch angle in radians.
        K: camera intrinsics.
        mode: interpolation mode for warping, default: 0 - 'linear', else 1 - 'nearest'
    
    Output:
        aligned_seg_map: gravity-aligned segmentation map.
    """
    # Validate input shape
    if seg_map.ndim != 3:
        raise ValueError("Segmentation map must be a 3D array with shape (N, H, W).")
    
    N, h, w = seg_map.shape
    
    # Calculate R_gc from roll and pitch
    p = -p  # This is because the pitch axis of robot and camera is in the opposite direction
    cr = np.cos(r)
    sr = np.sin(r)
    cp = np.cos(p)
    sp = np.sin(p)

    # Compute R_cg first
    R_x = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])  # Pitch
    R_z = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])  # Roll

    R_cg = R_z @ R_x
    R_gc = R_cg.T

    # Compute the homography matrix
    persp_M = K @ R_gc @ np.linalg.inv(K)

    # Create an empty array for the aligned segmentation map
    aligned_seg_map = np.zeros_like(seg_map)
    
    # Process each channel independently
    for i in range(N):
        # Align the current channel
        aligned_channel = cv2.warpPerspective(
            seg_map[i, :, :], persp_M, (w, h), flags=cv2.INTER_NEAREST if mode == 1 else cv2.INTER_LINEAR
        )
        aligned_seg_map[i, :, :] = aligned_channel
    
    return aligned_seg_map


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
    

def quaternion_to_euler(quaternions):
    """
    Convert an array of quaternions to Euler angles.
    
    Parameters:
        quaternions (np.ndarray): A 2D array where each row is a quaternion [q_w, q_x, q_y, q_z].
    
    Returns:
        euler_angles (np.ndarray): A 2D array where each row contains Euler angles [roll, pitch, yaw].
    """
    q_w = quaternions[:, 0]
    q_x = quaternions[:, 1]
    q_y = quaternions[:, 2]
    q_z = quaternions[:, 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (q_w * q_y - q_z * q_x)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * np.pi / 2,
                     np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    euler_angles = np.stack((roll, pitch, yaw), axis=-1)
    return euler_angles