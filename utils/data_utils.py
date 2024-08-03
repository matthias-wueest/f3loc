"""
Dataset for training structural depth prediction
"""

import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.utils import gravity_align, gravity_align_depth
from metric3d.hubconf import *
from src.depth_image_functions import *

class TrajDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        add_rp=False,
        roll=0,
        pitch=0,
        without_depth=False,
    ):
        super().__init__()
        """
        Contains L frames trajectories
        """
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []
        self.rgbs = []
        self.without_depth = without_depth
        self.load_scene_start_idx_and_depths_and_poses()
        self.N = self.scene_start_idx[-1]

    def __len__(self):
        return self.N

    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        for scene in self.scene_names:

            if not self.without_depth:
                # get depths
                if self.depth_dir == None:
                    depth_file = os.path.join(
                        self.dataset_dir, scene, self.depth_suffix + ".txt"
                    )
                else:
                    depth_file = os.path.join(
                        self.depth_dir, scene, self.depth_suffix + ".txt"
                    )

                # read depth
                with open(depth_file, "r") as f:
                    depths_txt = [line.strip() for line in f.readlines()]

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            traj_len = len(poses_txt)
            # traj_len has to be multiple of L
            traj_len -= traj_len % self.L
            scene_depths = []
            scene_poses = []
            for state_id in range(traj_len):
                if not self.without_depth:
                    # get depth
                    depth = depths_txt[state_id].split(" ")
                    depth = np.array([float(d) for d in depth]).astype(np.float32)
                    scene_depths.append(depth)

                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

            start_idx += traj_len // self.L
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)

    def __getitem__(self, idx):
        """
        data_dict:
            "ref_img": (3, H, W)
            "ref_pose": (3)
            "src_img": (L, 3, H, W)
            "src_pose": (L, 3)
            "ref_mask": (H, W)
            "src_mask": (L, H, W)
            "gt_depth": (fW), ground truth depth of the reference img
        """

        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        data_dict = {}
        if not self.without_depth:
            # get depths
            gt_depth = self.gt_depth[scene_idx][
                idx_within_scene * self.L : idx_within_scene * self.L + self.L
            ]
            data_dict["gt_depth"] = gt_depth

        # get poses
        poses = self.gt_pose[scene_idx][
            idx_within_scene * self.L : idx_within_scene * self.L + self.L
        ]
        data_dict["poses"] = poses

        # get images
        imgs = []
        for l in range(self.L):
            image_path = os.path.join(
                self.dataset_dir,
                scene_name,
                "rgb",
                str(idx_within_scene * self.L + l).zfill(5) + ".png",
            )
            img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
            imgs.append(img_l)
        imgs = np.stack(imgs, axis=0).astype(np.float32)  # (L, H, W, 3)

        if self.add_rp:
            # source
            masks = []
            for l in range(self.L):
                # get virtual roll and pitch
                r = (np.random.random() - 0.5) * 2 * self.roll
                p = (np.random.random() - 0.5) * 2 * self.pitch
                # generate mask
                mask = np.ones(list(imgs.shape[1:3]))
                mask = gravity_align(mask, r, p, mode=1)
                mask[mask < 1] = 0
                masks.append(mask.astype(np.uint8))
            masks = np.stack(masks, axis=0)  # (L, H, W)
            data_dict["masks"] = masks

            # source
            for l in range(self.L):
                # normalize => crop
                # normailize
                imgs[l, :, :, :] = (
                    cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
                )
                imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
                imgs[l, :, :, :] /= (0.229, 0.224, 0.225)

                # crop
                imgs[l, :, :, :][masks[l, :, :] == 0, :] = 0
        else:
            # source
            for l in range(self.L):
                # normalize => crop
                # normailize
                imgs[l, :, :, :] = (
                    cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
                )
                imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
                imgs[l, :, :, :] /= (0.229, 0.224, 0.225)

        # from H,W,C to C,H,W
        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)
        data_dict["imgs"] = imgs
        return data_dict



class TrajDataset_gibson_metric3d(Dataset):
    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        add_rp=False,
        roll=0,
        pitch=0,
        without_depth=False,
    ):
        super().__init__()
        """
        Contains L frames trajectories
        """
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []
        self.rgbs = []
        self.K=np.array([[240, 0, 320], [0, 240, 240], [0, 0, 1]])
        self.without_depth = without_depth
        self.load_scene_start_idx_and_depths_and_poses()
        self.N = self.scene_start_idx[-1]

    def __len__(self):
        return self.N

    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        for scene in self.scene_names:

            if not self.without_depth:
                # get depths
                if self.depth_dir == None:
                    depth_file = os.path.join(
                        self.dataset_dir, scene, self.depth_suffix + ".txt"
                    )
                else:
                    depth_file = os.path.join(
                        self.depth_dir, scene, self.depth_suffix + ".txt"
                    )

                # read depth
                with open(depth_file, "r") as f:
                    depths_txt = [line.strip() for line in f.readlines()]

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            traj_len = len(poses_txt)
            # traj_len has to be multiple of L
            traj_len -= traj_len % self.L
            scene_depths = []
            scene_poses = []
            for state_id in range(traj_len):
                if not self.without_depth:
                    # get depth
                    depth = depths_txt[state_id].split(" ")
                    depth = np.array([float(d) for d in depth]).astype(np.float32)
                    scene_depths.append(depth)

                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

            start_idx += traj_len // self.L
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)

    def __getitem__(self, idx):
        """
        data_dict:
            "ref_img": (3, H, W)
            "ref_pose": (3)
            "src_img": (L, 3, H, W)
            "src_pose": (L, 3)
            "ref_mask": (H, W)
            "src_mask": (L, H, W)
            "gt_depth": (fW), ground truth depth of the reference img
        """

        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        data_dict = {}
        if not self.without_depth:
            # get depths
            gt_depth = self.gt_depth[scene_idx][
                idx_within_scene * self.L : idx_within_scene * self.L + self.L
            ]
            data_dict["gt_depth"] = gt_depth

        # get poses
        poses = self.gt_pose[scene_idx][
            idx_within_scene * self.L : idx_within_scene * self.L + self.L
        ]
        data_dict["poses"] = poses

        # get images
        imgs = []
        for l in range(self.L):
            image_path = os.path.join(
                self.dataset_dir,
                scene_name,
                "rgb",
                str(idx_within_scene * self.L + l).zfill(5) + ".png",
            )
            img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
            imgs.append(img_l)
        imgs = np.stack(imgs, axis=0).astype(np.float32)  # (L, H, W, 3)

        if self.add_rp:
            # source
            masks = []
            for l in range(self.L):
                # get virtual roll and pitch
                r = (np.random.random() - 0.5) * 2 * self.roll
                p = (np.random.random() - 0.5) * 2 * self.pitch
                # generate mask
                mask = np.ones(list(imgs.shape[1:3]))
                mask = gravity_align(mask, r, p, mode=1)
                mask[mask < 1] = 0
                masks.append(mask.astype(np.uint8))
            masks = np.stack(masks, axis=0)  # (L, H, W)
            data_dict["masks"] = masks

            # source
            for l in range(self.L):
                # normalize => crop
                # normailize

                # ### For metric3D: do not normalize yet
                imgs[l, :, :, :] = (cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB))
                # imgs[l, :, :, :] = imgs[l, :, :, :] / 255.0
                # imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
                # imgs[l, :, :, :] /= (0.229, 0.224, 0.225)

                # crop
                # imgs[l, :, :, :][masks[l, :, :] == 0, :] = 0
        else:
            # source
            for l in range(self.L):
                # normalize => crop
                # normailize

                # ### For metric3D: do not normalize yet
                imgs[l, :, :, :] = (cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB))
                # imgs[l, :, :, :] = imgs[l, :, :, :] / 255.0
                # imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
                # imgs[l, :, :, :] /= (0.229, 0.224, 0.225)

        # get depth images
        depth_imgs = []
        for l in range(self.L):
            depth_img = get_depth_image_metric3d(imgs[l, :, :, :], K=self.K)
            depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array
            depth_imgs.append(depth_img)
            print("Depth image ", l, " done.")
        depth_imgs = np.stack(depth_imgs, axis=0).astype(np.float32)  # (L, H, W, 3)
        print(depth_imgs.shape)

        # # from H,W,C to C,H,W
        # imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)
        # data_dict["imgs"] = imgs

        data_dict["imgs"] = depth_imgs

        return data_dict




class TrajDataset_hge_customized_cropped(Dataset):
    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        add_rp=False,
        roll=0,
        pitch=0,
        without_depth=False,
    ):
        super().__init__()
        """
        Contains L frames trajectories
        """
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []
        self.rgbs = []
        self.without_depth = without_depth
        self.load_scene_start_idx_and_depths_and_poses()
        self.N = self.scene_start_idx[-1]

    def __len__(self):
        return self.N

    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        for scene in self.scene_names:

            if not self.without_depth:
                # get depths
                if self.depth_dir == None:
                    depth_file = os.path.join(
                        self.dataset_dir, scene, self.depth_suffix + ".txt"
                    )
                else:
                    depth_file = os.path.join(
                        self.depth_dir, scene, self.depth_suffix + ".txt"
                    )

                # read depth
                with open(depth_file, "r") as f:
                    depths_txt = [line.strip() for line in f.readlines()]

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            traj_len = len(poses_txt)
            # traj_len has to be multiple of L
            traj_len -= traj_len % self.L
            scene_depths = []
            scene_poses = []
            for state_id in range(traj_len):
                if not self.without_depth:
                    # get depth
                    depth = depths_txt[state_id].split(" ")
                    depth = np.array([float(d) for d in depth]).astype(np.float32)
                    scene_depths.append(depth)

                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

            start_idx += traj_len // self.L
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)

    def __getitem__(self, idx):
        """
        data_dict:
            "ref_img": (3, H, W)
            "ref_pose": (3)
            "src_img": (L, 3, H, W)
            "src_pose": (L, 3)
            "ref_mask": (H, W)
            "src_mask": (L, H, W)
            "gt_depth": (fW), ground truth depth of the reference img
        """

        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        data_dict = {}
        if not self.without_depth:
            # get depths
            gt_depth = self.gt_depth[scene_idx][
                idx_within_scene * self.L : idx_within_scene * self.L + self.L
            ]
            data_dict["gt_depth"] = gt_depth

        # get poses
        poses = self.gt_pose[scene_idx][
            idx_within_scene * self.L : idx_within_scene * self.L + self.L
        ]
        data_dict["poses"] = poses

        # get images
        imgs = []
        for l in range(self.L):
            image_path = os.path.join(
                self.dataset_dir,
                scene_name,
                "rgb",
                #str(idx_within_scene * self.L + l).zfill(5) + ".png",
                str(idx_within_scene * self.L + l).zfill(5) + "-0" + ".jpg",
            )
#            img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
#            imgs.append(img_l)
            imgs.append(5)
        imgs = np.stack(imgs, axis=0).astype(np.float32)  # (L, H, W, 3)

        if self.add_rp:
            # source
            masks = []
            for l in range(self.L):
                # get virtual roll and pitch
                r = (np.random.random() - 0.5) * 2 * self.roll
                p = (np.random.random() - 0.5) * 2 * self.pitch
                # generate mask
                mask = np.ones(list(imgs.shape[1:3]))
                mask = gravity_align(mask, r, p, mode=1)
                mask[mask < 1] = 0
                masks.append(mask.astype(np.uint8))
            masks = np.stack(masks, axis=0)  # (L, H, W)
            data_dict["masks"] = masks

            # source
            for l in range(self.L):
                # normalize => crop
                # normailize
                imgs[l, :, :, :] = (
                    cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
                )
                imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
                imgs[l, :, :, :] /= (0.229, 0.224, 0.225)

                # crop
                imgs[l, :, :, :][masks[l, :, :] == 0, :] = 0
#        else:
#            # source
#            for l in range(self.L):
#                # normalize => crop
#                # normailize
#                imgs[l, :, :, :] = (
#                    cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
#                )
#                imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
#                imgs[l, :, :, :] /= (0.229, 0.224, 0.225)

        # from H,W,C to C,H,W
#        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)
        data_dict["imgs"] = imgs
        return data_dict



class TrajDataset_hge_customized_cropped_gravity_align(Dataset):
    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        add_rp=False,
        roll=0,
        pitch=0,
        without_depth=False,
    ):
        super().__init__()
        """
        Contains L frames trajectories
        """
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []
        self.rgbs = []
        self.without_depth = without_depth
        self.K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])
        self.gt_euler_angles = []
        self.load_scene_start_idx_and_depths_and_poses()
        self.N = self.scene_start_idx[-1]

    def __len__(self):
        return self.N

    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        print("self.scene_names: ", self.scene_names)
        for scene in self.scene_names:
            print("scene: ", scene)
            if not self.without_depth:
                # get depths
                if self.depth_dir == None:
                    depth_file = os.path.join(
                        self.dataset_dir, scene, self.depth_suffix + ".txt"
                    )
                else:
                    depth_file = os.path.join(
                        self.depth_dir, scene, self.depth_suffix + ".txt"
                    )

                # read depth
                with open(depth_file, "r") as f:
                    depths_txt = [line.strip() for line in f.readlines()]

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            # read euler angles
            euler_angles_file = os.path.join(self.dataset_dir, scene, "euler_angles.txt")
            with open(euler_angles_file, "r") as f:
                euler_angles_txt = [line.strip() for line in f.readlines()]
            
            traj_len = len(poses_txt)
            # traj_len has to be multiple of L
            traj_len -= traj_len % self.L
            scene_depths = []
            scene_poses = []
            scene_euler_angles = []
            for state_id in range(traj_len):
                if not self.without_depth:
                    # get depth
                    depth = depths_txt[state_id].split(" ")
                    depth = np.array([float(d) for d in depth]).astype(np.float32)
                    scene_depths.append(depth)

                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

                # get roll and pitch
                euler_angles = euler_angles_txt[state_id].split(" ")
                euler_angles = np.array([float(s) for s in euler_angles]).astype(np.float32)
                scene_euler_angles.append(euler_angles)

            start_idx += traj_len // self.L
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)
            self.gt_euler_angles.append(scene_euler_angles)

    def __getitem__(self, idx):
        """
        data_dict:
            "ref_img": (3, H, W)
            "ref_pose": (3)
            "src_img": (L, 3, H, W)
            "src_pose": (L, 3)
            "ref_mask": (H, W)
            "src_mask": (L, H, W)
            "gt_depth": (fW), ground truth depth of the reference img
        """

        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        data_dict = {}
        if not self.without_depth:
            # get depths
            gt_depth = self.gt_depth[scene_idx][
                idx_within_scene * self.L : idx_within_scene * self.L + self.L
            ]
            data_dict["gt_depth"] = gt_depth

        # get poses
        poses = self.gt_pose[scene_idx][
            idx_within_scene * self.L : idx_within_scene * self.L + self.L
        ]
        data_dict["poses"] = poses

        # get reference euler angles
        ref_euler_angles = self.gt_euler_angles[scene_idx][
            idx_within_scene * self.L : idx_within_scene * self.L + self.L
        ]
        data_dict["euler_angles"] = ref_euler_angles

        # get images
        imgs = []
        for l in range(self.L):
            image_path = os.path.join(
                self.dataset_dir,
                scene_name,
                "rgb",
                #str(idx_within_scene * self.L + l).zfill(5) + ".png",
                str(idx_within_scene * self.L + l).zfill(5) + "-0" + ".jpg",
            )
            img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
            imgs.append(img_l)
#            imgs.append(5)
        imgs = np.stack(imgs, axis=0).astype(np.float32)  # (L, H, W, 3)

###

        masks = []
        for l in range(self.L):
            roll = ref_euler_angles[l][0]
            pitch = ref_euler_angles[l][1]
            # align image to gravity
            imgs[l, :, :, :] = gravity_align(imgs[l, :, :, :], r=pitch, p=-(roll+np.pi/2), mode=1, K=self.K)
            # generate mask
            mask = np.ones(list(imgs[l, :, :, :].shape[:2]))
            #mask = gravity_align(mask, r, p, mode=1)
            mask = gravity_align(mask, r=pitch, p=-(roll+np.pi/2),  mode=1, K=self.K)
            mask[mask < 1] = 0
            masks.append(mask.astype(np.uint8))
        masks = np.stack(masks, axis=0)  # (L, H, W)
        data_dict["masks"] = masks

        for l in range(self.L):
            # normalize => crop
            # normalize
            imgs[l, :, :, :] = cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
            imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
            imgs[l, :, :, :] /= (0.229, 0.224, 0.225)
            # crop
            #print(type(imgs))
            #print(type(masks))
            imgs[l, :, :, :][masks[l, :, :] == 0, :] = 0
        
###

        if False: #self.add_rp:
            # source
            masks = []
            for l in range(self.L):
                # get virtual roll and pitch
                r = (np.random.random() - 0.5) * 2 * self.roll
                p = (np.random.random() - 0.5) * 2 * self.pitch
                # generate mask
                mask = np.ones(list(imgs.shape[1:3]))
                mask = gravity_align(mask, r, p, mode=1)
                mask[mask < 1] = 0
                masks.append(mask.astype(np.uint8))
            masks = np.stack(masks, axis=0)  # (L, H, W)
            data_dict["masks"] = masks

            # source
            for l in range(self.L):
                # normalize => crop
                # normailize
                imgs[l, :, :, :] = (
                    cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
                )
                imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
                imgs[l, :, :, :] /= (0.229, 0.224, 0.225)

                # crop
                imgs[l, :, :, :][masks[l, :, :] == 0, :] = 0
        elif False: #else:
            # source
            for l in range(self.L):
                # normalize => crop
                # normailize
                imgs[l, :, :, :] = (
                    cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
                )
                imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
                imgs[l, :, :, :] /= (0.229, 0.224, 0.225)

        # from H,W,C to C,H,W
        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)
        data_dict["imgs"] = imgs

        return data_dict



class TrajDataset_hge_customized_metric3d(Dataset):
    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        add_rp=False,
        roll=0,
        pitch=0,
        without_depth=False,
    ):
        super().__init__()
        """
        Contains L frames trajectories
        """
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []
        self.rgbs = []
        self.K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])
        self.gt_euler_angles = []
        self.without_depth = without_depth
        self.load_scene_start_idx_and_depths_and_poses()
        self.N = self.scene_start_idx[-1]

    def __len__(self):
        return self.N

    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        for scene in self.scene_names:

            if not self.without_depth:
                # get depths
                if self.depth_dir == None:
                    depth_file = os.path.join(
                        self.dataset_dir, scene, self.depth_suffix + ".txt"
                    )
                else:
                    depth_file = os.path.join(
                        self.depth_dir, scene, self.depth_suffix + ".txt"
                    )

                # read depth
                with open(depth_file, "r") as f:
                    depths_txt = [line.strip() for line in f.readlines()]

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            # read euler angles
            euler_angles_file = os.path.join(self.dataset_dir, scene, "euler_angles.txt")
            with open(euler_angles_file, "r") as f:
                euler_angles_txt = [line.strip() for line in f.readlines()]

            traj_len = len(poses_txt)
            # traj_len has to be multiple of L
            traj_len -= traj_len % self.L
            scene_depths = []
            scene_poses = []
            scene_euler_angles = []
            for state_id in range(traj_len):
                if not self.without_depth:
                    # get depth
                    depth = depths_txt[state_id].split(" ")
                    depth = np.array([float(d) for d in depth]).astype(np.float32)
                    scene_depths.append(depth)

                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

                # get roll and pitch
                euler_angles = euler_angles_txt[state_id].split(" ")
                euler_angles = np.array([float(s) for s in euler_angles]).astype(np.float32)
                scene_euler_angles.append(euler_angles)

            start_idx += traj_len // self.L
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)
            self.gt_euler_angles.append(scene_euler_angles)

    def __getitem__(self, idx):
        """
        data_dict:
            "ref_img": (3, H, W)
            "ref_pose": (3)
            "src_img": (L, 3, H, W)
            "src_pose": (L, 3)
            "ref_mask": (H, W)
            "src_mask": (L, H, W)
            "gt_depth": (fW), ground truth depth of the reference img
        """

        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        data_dict = {}
        if not self.without_depth:
            # get depths
            gt_depth = self.gt_depth[scene_idx][
                idx_within_scene * self.L : idx_within_scene * self.L + self.L
            ]
            data_dict["gt_depth"] = gt_depth

        # get poses
        poses = self.gt_pose[scene_idx][
            idx_within_scene * self.L : idx_within_scene * self.L + self.L
        ]
        data_dict["poses"] = poses

        # get reference euler angles
        ref_euler_angles = self.gt_euler_angles[scene_idx][
            idx_within_scene * self.L : idx_within_scene * self.L + self.L
        ]
        data_dict["euler_angles"] = ref_euler_angles

        # get images
        imgs = []
        for l in range(self.L):
            image_path = os.path.join(
                self.dataset_dir,
                scene_name,
                "rgb",
                #str(idx_within_scene * self.L + l).zfill(5) + ".png",
                str(idx_within_scene * self.L + l).zfill(5) + "-0" + ".jpg",
            )
            img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
            imgs.append(img_l)
            #imgs.append(5)
        imgs = np.stack(imgs, axis=0).astype(np.float32)  # (L, H, W, 3)

        if self.add_rp:
            # source
            masks = []
            for l in range(self.L):
                # get virtual roll and pitch
                r = (np.random.random() - 0.5) * 2 * self.roll
                p = (np.random.random() - 0.5) * 2 * self.pitch
                # generate mask
                mask = np.ones(list(imgs.shape[1:3]))
                mask = gravity_align(mask, r, p, mode=1)
                mask[mask < 1] = 0
                masks.append(mask.astype(np.uint8))
            masks = np.stack(masks, axis=0)  # (L, H, W)
            data_dict["masks"] = masks

            # source
            for l in range(self.L):
                # normalize => crop
                # normalize
                
                # ### For metric3D: do not normalize yet
                imgs[l, :, :, :] = cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB)
                # imgs[l, :, :, :] = imgs[l, :, :, :] / 255.0
                # imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
                # imgs[l, :, :, :] /= (0.229, 0.224, 0.225)
            
                # crop
                #imgs[l, :, :, :][masks[l, :, :] == 0, :] = 0
            ###
        # TODO: BGR => RGB
        # else:
        #     # source
        #     for l in range(self.L):
        #         # normalize => crop
        #         # normailize
# 
        #         # ### For metric3D: do not normalize yet
        #         imgs[l, :, :, :] = cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB)
        #         # imgs[l, :, :, :] = imgs[l, :, :, :] / 255.0
        #         # imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
        #         # imgs[l, :, :, :] /= (0.229, 0.224, 0.225)


        # get depth images
        depth_imgs = []
        for l in range(self.L):
            depth_img = get_depth_image_metric3d(imgs[l, :, :, :])#, K=self.K)
            depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array
            depth_imgs.append(depth_img)
            print("Depth image ", l, " done.")
        depth_imgs = np.stack(depth_imgs, axis=0).astype(np.float32)  # (L, H, W, 3)
        print(depth_imgs.shape)

        # gravity align depth images
        for l in range(self.L):
            r=ref_euler_angles[l][1]
            p=-(ref_euler_angles[l][0]+np.pi/2)
            depth_imgs[l, :, :] = gravity_align_depth(depth_imgs[l, :, :], r, p, K=self.K)


#        data_dict["depth_imgs"] = depth_imgs

#        else:
#            # source
#            for l in range(self.L):
#                # normalize => crop
#                # normailize
#                imgs[l, :, :, :] = (
#                    cv2.cvtColor(imgs[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
#                )
#                imgs[l, :, :, :] -= (0.485, 0.456, 0.406)
#                imgs[l, :, :, :] /= (0.229, 0.224, 0.225)

        # from H,W,C to C,H,W
#        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)
#        data_dict["imgs"] = imgs
        data_dict["imgs"] = depth_imgs

        return data_dict




class GridSeqDataset(Dataset):

    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        add_rp=False,
        roll=0,
        pitch=0,
        start_scene=None,
        end_scene=None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.start_scene = start_scene
        self.end_scene = end_scene
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []
        self.rgbs = []
        self.load_scene_start_idx_and_depths_and_poses()
        if start_scene is None:
            self.N = self.scene_start_idx[-1]
        else:
            # compute N
            self.N = (
                self.scene_start_idx[self.end_scene + 1]
                - self.scene_start_idx[self.start_scene]
            )

    def __len__(self):
        return self.N

    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        for scene in self.scene_names:

            # get depths
            if self.depth_dir == None:
                depth_file = os.path.join(
                    self.dataset_dir, scene, self.depth_suffix + ".txt"
                )
            else:
                depth_file = os.path.join(
                    self.depth_dir, scene, self.depth_suffix + ".txt"
                )

            # read depth
            with open(depth_file, "r") as f:
                depths_txt = [line.strip() for line in f.readlines()]

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            traj_len = len(poses_txt)
            scene_depths = []
            scene_poses = []
            for state_id in range(traj_len):
                # get depth
                depth = depths_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth]).astype(np.float32)
                scene_depths.append(depth)

                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

            start_idx += traj_len // (self.L + 1)
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)

    def __getitem__(self, idx):
        """
        data_dict:
            "ref_img": (3, H, W)
            "ref_pose": (3)
            "src_img": (L, 3, H, W)
            "src_pose": (L, 3)
            "ref_mask": (H, W)
            "src_mask": (L, H, W)
            "ref_depth": (fW), ground truth depth of the reference img
            "src_depth": (L, fW), ground truth depth of the source img
        """
        if self.start_scene is not None:
            idx += self.scene_start_idx[self.start_scene]

        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # get reference depth
        ref_depth = self.gt_depth[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict = {"ref_depth": ref_depth}

        # get source depth
        src_depth = np.stack(
            self.gt_depth[scene_idx][
                idx_within_scene * (self.L + 1) : idx_within_scene * (self.L + 1)
                + self.L
            ],
            axis=0,
        )
        data_dict["src_depth"] = src_depth

        # get reference pose
        ref_pose = self.gt_pose[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict["ref_noise"] = 0
        data_dict["ref_pose"] = ref_pose

        # get source pose
        src_pose = np.stack(
            self.gt_pose[scene_idx][
                idx_within_scene * (self.L + 1) : idx_within_scene * (self.L + 1)
                + self.L
            ],
            axis=0,
        )
        data_dict["src_noise"] = 0
        data_dict["src_pose"] = src_pose

        # get source images
        src_img = []
        for l in range(self.L):
            image_path = os.path.join(
                self.dataset_dir,
                scene_name,
                "rgb",
                str(idx_within_scene).zfill(5) + "-" + str(l) + ".png",
            )
            src_img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
            src_img.append(src_img_l)
        src_img = np.stack(src_img, axis=0).astype(np.float32)  # (L, H, W, 3)
        # get reference image
        image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "rgb",
            str(idx_within_scene).zfill(5) + "-" + str(self.L) + ".png",
        )
        ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(
            np.float32
        )  # (H, W, 3)

        if self.add_rp:
            # reference
            # get virtual roll and pitch
            r = (np.random.random() - 0.5) * 2 * self.roll
            p = (np.random.random() - 0.5) * 2 * self.pitch
            # generate mask
            mask = np.ones(list(ref_img.shape[:2]))
            #mask = gravity_align(mask, r, p, visualize=False, mode=1)
            mask = gravity_align(mask, r, p, mode=1)
            mask[mask < 1] = 0
            ref_mask = mask.astype(np.uint8)
            data_dict["ref_mask"] = ref_mask

            # source
            src_mask = []
            for l in range(self.L):
                # get virtual roll and pitch
                r = (np.random.random() - 0.5) * 2 * self.roll
                p = (np.random.random() - 0.5) * 2 * self.pitch
                # generate mask
                mask = np.ones(list(ref_img.shape[:2]))
                #mask = gravity_align(mask, r, p, visualize=False, mode=1)
                mask = gravity_align(mask, r, p, mode=1)
                mask[mask < 1] = 0
                src_mask.append(mask.astype(np.uint8))
            src_mask = np.stack(src_mask, axis=0)  # (L, H, W)
            data_dict["src_mask"] = src_mask

            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)

            # crop
            ref_img[ref_mask == 0, :] = 0

            # source
            for l in range(self.L):
                # normalize => crop
                # normailize
                src_img[l, :, :, :] = (
                    cv2.cvtColor(src_img[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
                )
                src_img[l, :, :, :] -= (0.485, 0.456, 0.406)
                src_img[l, :, :, :] /= (0.229, 0.224, 0.225)

                # crop
                src_img[l, :, :, :][src_mask[l, :, :] == 0, :] = 0
        else:
            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)

            # source
            for l in range(self.L):
                # normalize => crop
                # normailize
                src_img[l, :, :, :] = (
                    cv2.cvtColor(src_img[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
                )
                src_img[l, :, :, :] -= (0.485, 0.456, 0.406)
                src_img[l, :, :, :] /= (0.229, 0.224, 0.225)

        # from H,W,C to C,H,W
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
        src_img = np.transpose(src_img, (0, 3, 1, 2)).astype(np.float32)
        data_dict["ref_img"] = ref_img
        data_dict["src_img"] = src_img

        return data_dict



class GridSeqDataset_hge_customized_cropped(Dataset):

    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        add_rp=False,
        roll=0,
        pitch=0,
        start_scene=None,
        end_scene=None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.start_scene = start_scene
        self.end_scene = end_scene
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []
        self.rgbs = []
        self.load_scene_start_idx_and_depths_and_poses()
        if start_scene is None:
            self.N = self.scene_start_idx[-1]
        else:
            # compute N
            self.N = (
                self.scene_start_idx[self.end_scene + 1]
                - self.scene_start_idx[self.start_scene]
            )

    def __len__(self):
        return self.N

    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        for scene in self.scene_names:

            # get depths
            if self.depth_dir == None:
                depth_file = os.path.join(
                    self.dataset_dir, scene, self.depth_suffix + ".txt"
                )
            else:
                depth_file = os.path.join(
                    self.depth_dir, scene, self.depth_suffix + ".txt"
                )

            # read depth
            with open(depth_file, "r") as f:
                depths_txt = [line.strip() for line in f.readlines()]

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            traj_len = len(poses_txt)
            scene_depths = []
            scene_poses = []
            for state_id in range(traj_len):
                # get depth
                depth = depths_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth]).astype(np.float32)
                scene_depths.append(depth)

                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

            start_idx += traj_len // (self.L + 1)
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)

    def __getitem__(self, idx):
        """
        data_dict:
            "ref_img": (3, H, W)
            "ref_pose": (3)
            "src_img": (L, 3, H, W)
            "src_pose": (L, 3)
            "ref_mask": (H, W)
            "src_mask": (L, H, W)
            "ref_depth": (fW), ground truth depth of the reference img
            "src_depth": (L, fW), ground truth depth of the source img
        """
        if self.start_scene is not None:
            idx += self.scene_start_idx[self.start_scene]

        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # get reference depth
        ref_depth = self.gt_depth[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict = {"ref_depth": ref_depth}

    #     # get source depth
    #     src_depth = np.stack(
    #         self.gt_depth[scene_idx][
    #             idx_within_scene * (self.L + 1) : idx_within_scene * (self.L + 1)
    #             + self.L
    #         ],
    #         axis=0,
    #     )
    #     data_dict["src_depth"] = src_depth

        # get reference pose
        ref_pose = self.gt_pose[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict["ref_noise"] = 0
        data_dict["ref_pose"] = ref_pose

    #     # get source pose
    #     src_pose = np.stack(
    #         self.gt_pose[scene_idx][
    #             idx_within_scene * (self.L + 1) : idx_within_scene * (self.L + 1)
    #             + self.L
    #         ],
    #         axis=0,
    #     )
    #     data_dict["src_noise"] = 0
    #     data_dict["src_pose"] = src_pose

    #     # get source images
    #     src_img = []
    #     for l in range(self.L):
    #         image_path = os.path.join(
    #             self.dataset_dir,
    #             scene_name,
    #             "rgb",
    #             str(idx_within_scene).zfill(5) + "-" + str(l) + ".png",
    #         )
    #         src_img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #         src_img.append(src_img_l)
    #     src_img = np.stack(src_img, axis=0).astype(np.float32)  # (L, H, W, 3)


        # get reference image
        image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "rgb",
            #str(idx_within_scene).zfill(5) + "-" + str(self.L) + ".png",
            str(idx_within_scene).zfill(5) + "-0" + ".jpg",
        )
        ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(
            np.float32
        )  # (H, W, 3)


        if self.add_rp:
            # reference
            # get virtual roll and pitch
            r = (np.random.random() - 0.5) * 2 * self.roll
            p = (np.random.random() - 0.5) * 2 * self.pitch
            # generate mask
            mask = np.ones(list(ref_img.shape[:2]))
            #mask = gravity_align(mask, r, p, visualize=False, mode=1)
            mask = gravity_align(mask, r, p, mode=1)
            mask[mask < 1] = 0
            ref_mask = mask.astype(np.uint8)
            data_dict["ref_mask"] = ref_mask

    #         # source
    #         src_mask = []
    #         for l in range(self.L):
    #             # get virtual roll and pitch
    #             r = (np.random.random() - 0.5) * 2 * self.roll
    #             p = (np.random.random() - 0.5) * 2 * self.pitch
    #             # generate mask
    #             mask = np.ones(list(ref_img.shape[:2]))
    #             #mask = gravity_align(mask, r, p, visualize=False, mode=1)
    #             mask = gravity_align(mask, r, p, mode=1)
    #             mask[mask < 1] = 0
    #             src_mask.append(mask.astype(np.uint8))
    #         src_mask = np.stack(src_mask, axis=0)  # (L, H, W)
    #         data_dict["src_mask"] = src_mask

            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)

            # crop
            ref_img[ref_mask == 0, :] = 0

    #         # source
    #         for l in range(self.L):
    #             # normalize => crop
    #             # normailize
    #             src_img[l, :, :, :] = (
    #                 cv2.cvtColor(src_img[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
    #             )
    #             src_img[l, :, :, :] -= (0.485, 0.456, 0.406)
    #             src_img[l, :, :, :] /= (0.229, 0.224, 0.225)
# 
    #             # crop
    #             src_img[l, :, :, :][src_mask[l, :, :] == 0, :] = 0
        else:
            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)

    #         # source
    #         for l in range(self.L):
    #             # normalize => crop
    #             # normailize
    #             src_img[l, :, :, :] = (
    #                 cv2.cvtColor(src_img[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
    #             )
    #             src_img[l, :, :, :] -= (0.485, 0.456, 0.406)
    #             src_img[l, :, :, :] /= (0.229, 0.224, 0.225)

        # from H,W,C to C,H,W
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
    #    src_img = np.transpose(src_img, (0, 3, 1, 2)).astype(np.float32)
        data_dict["ref_img"] = ref_img
    #    data_dict["src_img"] = src_img

        return data_dict



class GridSeqDataset_hge_customized_cropped_gravity_align(Dataset):

    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        add_rp=False,
        roll=0,
        pitch=0,
        start_scene=None,
        end_scene=None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.start_scene = start_scene
        self.end_scene = end_scene
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []
        self.rgbs = []
        self.K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])
        self.gt_euler_angles = []
        self.load_scene_start_idx_and_depths_and_poses()
        if start_scene is None:
            self.N = self.scene_start_idx[-1]
        else:
            # compute N
            self.N = (
                self.scene_start_idx[self.end_scene + 1]
                - self.scene_start_idx[self.start_scene]
            )

    def __len__(self):
        return self.N

    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        print("self.scene_names: ", self.scene_names)
        for scene in self.scene_names:
            print("scene: ", scene)
            # get depths
            if self.depth_dir == None:
                depth_file = os.path.join(
                    self.dataset_dir, scene, self.depth_suffix + ".txt"
                )
            else:
                depth_file = os.path.join(
                    self.depth_dir, scene, self.depth_suffix + ".txt"
                )

            # read depth
            with open(depth_file, "r") as f:
                depths_txt = [line.strip() for line in f.readlines()]

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            # read euler angles
            euler_angles_file = os.path.join(self.dataset_dir, scene, "euler_angles.txt")
            with open(euler_angles_file, "r") as f:
                euler_angles_txt = [line.strip() for line in f.readlines()]

            traj_len = len(poses_txt)
            scene_depths = []
            scene_poses = []
            scene_euler_angles = []
            for state_id in range(traj_len):
                # get depth
                depth = depths_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth]).astype(np.float32)
                scene_depths.append(depth)

                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

                # get roll and pitch
                euler_angles = euler_angles_txt[state_id].split(" ")
                euler_angles = np.array([float(s) for s in euler_angles]).astype(np.float32)
                scene_euler_angles.append(euler_angles)
                #print("euler_angles: ", euler_angles)

            #print("scene_euler_angles: ", scene_euler_angles)

            start_idx += traj_len // (self.L + 1)
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)
            self.gt_euler_angles.append(scene_euler_angles)


    def __getitem__(self, idx):
        """
        data_dict:
            "ref_img": (3, H, W)
            "ref_pose": (3)
            "src_img": (L, 3, H, W)
            "src_pose": (L, 3)
            "ref_mask": (H, W)
            "src_mask": (L, H, W)
            "ref_depth": (fW), ground truth depth of the reference img
            "src_depth": (L, fW), ground truth depth of the source img
        """
        if self.start_scene is not None:
            idx += self.scene_start_idx[self.start_scene]

        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # get reference depth
        ref_depth = self.gt_depth[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict = {"ref_depth": ref_depth}

        # get reference pose
        ref_pose = self.gt_pose[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict["ref_noise"] = 0
        data_dict["ref_pose"] = ref_pose

        # get reference euler angles
        ref_euler_angles = self.gt_euler_angles[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict["euler_angles"] = ref_euler_angles

        # get reference image
        image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "rgb",
            #str(idx_within_scene).zfill(5) + "-" + str(self.L) + ".png",
            str(idx_within_scene).zfill(5) + "-0" + ".jpg",
        )
        ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(
            np.float32
        )  # (H, W, 3)

        
        # align image to gravity
        roll = ref_euler_angles[0]
        pitch = ref_euler_angles[1]
        ref_img = gravity_align(ref_img, r=pitch, p=-(roll+np.pi/2),  mode=1, K=self.K)

        # generate mask
        mask = np.ones(list(ref_img.shape[:2]))
        #mask = gravity_align(mask, r, p, mode=1)
        mask = gravity_align(mask, r=pitch, p=-(roll+np.pi/2),  mode=1, K=self.K)
        mask[mask < 1] = 0
        ref_mask = mask.astype(np.uint8)
        data_dict["ref_mask"] = ref_mask

        # normalize => crop
        # normalize
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
        ref_img -= (0.485, 0.456, 0.406)
        ref_img /= (0.229, 0.224, 0.225)
        # crop
        ref_img[ref_mask == 0, :] = 0


        if False: #self.add_rp:
            # reference
            # get virtual roll and pitch
            r = (np.random.random() - 0.5) * 2 * self.roll
            p = (np.random.random() - 0.5) * 2 * self.pitch
            # generate mask
            mask = np.ones(list(ref_img.shape[:2]))
            #mask = gravity_align(mask, r, p, visualize=False, mode=1)
            mask = gravity_align(mask, r, p, mode=1)
            mask[mask < 1] = 0
            ref_mask = mask.astype(np.uint8)
            data_dict["ref_mask"] = ref_mask

            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)

            # crop
            ref_img[ref_mask == 0, :] = 0

        elif False: #else:
            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)


        # from H,W,C to C,H,W
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
        #print(f"Final ref_img shape: {ref_img.shape}")  # Debug statement
        data_dict["ref_img"] = ref_img
        
        data_dict["scene_name"] = scene_name

        return data_dict


class GridSeqDataset_hge_customized_cropped_metric3d(Dataset):

    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        add_rp=False,
        roll=0,
        pitch=0,
        start_scene=None,
        end_scene=None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.start_scene = start_scene
        self.end_scene = end_scene
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []
        self.rgbs = []
        self.K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])
        self.gt_euler_angles = []
        self.metric3d_input_size = (616, 1064) # for vit model
        self.metric3d_model_name = "ViT-Small" #"ViT-Large"
        self.min_depth = 0.0
        self.max_depth = 100.0
        self.load_scene_start_idx_and_depths_and_poses()
        if start_scene is None:
            self.N = self.scene_start_idx[-1]
        else:
            # compute N
            self.N = (
                self.scene_start_idx[self.end_scene + 1]
                - self.scene_start_idx[self.start_scene]
            )

    def __len__(self):
        return self.N

    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        print("self.scene_names: ", self.scene_names)
        for scene in self.scene_names:
            print("scene: ", scene)
            # get depths
            if self.depth_dir == None:
                depth_file = os.path.join(
                    self.dataset_dir, scene, self.depth_suffix + ".txt"
                )
            else:
                depth_file = os.path.join(
                    self.depth_dir, scene, self.depth_suffix + ".txt"
                )

            # read depth
            with open(depth_file, "r") as f:
                depths_txt = [line.strip() for line in f.readlines()]

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            # read euler angles
            euler_angles_file = os.path.join(self.dataset_dir, scene, "euler_angles.txt")
            with open(euler_angles_file, "r") as f:
                euler_angles_txt = [line.strip() for line in f.readlines()]

            traj_len = len(poses_txt)
            scene_depths = []
            scene_poses = []
            scene_euler_angles = []
            for state_id in range(traj_len):
                # get depth
                depth = depths_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth]).astype(np.float32)
                scene_depths.append(depth)

                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

                # get roll and pitch
                euler_angles = euler_angles_txt[state_id].split(" ")
                euler_angles = np.array([float(s) for s in euler_angles]).astype(np.float32)
                scene_euler_angles.append(euler_angles)
                #print("euler_angles: ", euler_angles)

            #print("scene_euler_angles: ", scene_euler_angles)

            start_idx += traj_len // (self.L + 1)
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)
            self.gt_euler_angles.append(scene_euler_angles)


    def __getitem__(self, idx):
        """
        data_dict:
            "ref_img": (3, H, W)
            "ref_pose": (3)
            "src_img": (L, 3, H, W)
            "src_pose": (L, 3)
            "ref_mask": (H, W)
            "src_mask": (L, H, W)
            "ref_depth": (fW), ground truth depth of the reference img
            "src_depth": (L, fW), ground truth depth of the source img
        """
        if self.start_scene is not None:
            idx += self.scene_start_idx[self.start_scene]

        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # get reference depth
        ref_depth = self.gt_depth[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict = {"ref_depth": ref_depth}

        # get reference pose
        ref_pose = self.gt_pose[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict["ref_noise"] = 0
        data_dict["ref_pose"] = ref_pose

        # get reference euler angles
        ref_euler_angles = self.gt_euler_angles[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict["euler_angles"] = ref_euler_angles

        # get reference image
        image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "rgb",
            #str(idx_within_scene).zfill(5) + "-" + str(self.L) + ".png",
            str(idx_within_scene).zfill(5) + "-0" + ".jpg",
        )
        #ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(
        #    np.float32
        #)  # (H, W, 3)
        rgb_file = image_path
        rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]
        #print(f"Original RGB shape: {rgb_origin.shape}")  # Debug statement

        model_name = self.metric3d_model_name
        intrinsic = [self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]] #self.K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])
        
        #### adjust input size to fit pretrained model
        # keep ratio resize
        input_size = self.metric3d_input_size
        # input_size = (544, 1216) # for convnext model
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        #print(f"Resized RGB shape: {rgb.shape}")  # Debug statement
        # remember to scale intrinsic, hold depth
        intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        #print(f"Padded RGB shape: {rgb.shape}")  # Debug statement
        #### normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()
        #print(f"Normalized RGB shape: {rgb.shape}")  # Debug statement
        ###################### canonical camera space ######################        
        # inference
        if model_name == "ViT-Small":
            model = metric3d_vit_small(pretrain=True)
        elif model_name == "ViT-Large":
            model = metric3d_vit_large(pretrain=True)
        elif model_name == "ViT-giant2":
            model = metric3d_vit_giant2(pretrain=True)
        model.cuda().eval()
        #start_time = time.time()
        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({'input': rgb})
        #end_time = time.time()
        #execution_time = end_time - start_time
        #print(f"Execution time: {execution_time} seconds")

        # un pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        
        # upsample to original size
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
        ###################### canonical camera space ######################

        #### de-canonical transform
        canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300)
        ref_img = pred_depth.cpu().numpy()

        #### align image to gravity
        roll = ref_euler_angles[0]
        pitch = ref_euler_angles[1]
        ref_img = gravity_align(ref_img, r=pitch, p=-(roll+np.pi/2),  mode=1, K=self.K)

        # generate mask
        mask = np.ones(list(ref_img.shape[:2]))
        #mask = gravity_align(mask, r, p, mode=1)
        mask = gravity_align(mask, r=pitch, p=-(roll+np.pi/2),  mode=1, K=self.K)
        mask[mask < 1] = 0
        ref_mask = mask.astype(np.uint8)
        data_dict["ref_mask"] = ref_mask

        # normalize => crop
        # normalize
        #ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
        #ref_img -= (0.485, 0.456, 0.406)
        #ref_img /= (0.229, 0.224, 0.225)
        ref_img = (ref_img - self.min_depth) / (self.max_depth - self.min_depth)

        # crop
        ref_img[ref_mask == 0,] = 0
        ref_img = np.expand_dims(ref_img, axis=0)
        #ref_img[ref_mask == 0, :] = 0
        #print(f"Final ref_img shape: {ref_img.shape}")  # Debug statement


        if False: #self.add_rp:
            # reference
            # get virtual roll and pitch
            r = (np.random.random() - 0.5) * 2 * self.roll
            p = (np.random.random() - 0.5) * 2 * self.pitch
            # generate mask
            mask = np.ones(list(ref_img.shape[:2]))
            #mask = gravity_align(mask, r, p, visualize=False, mode=1)
            mask = gravity_align(mask, r, p, mode=1)
            mask[mask < 1] = 0
            ref_mask = mask.astype(np.uint8)
            data_dict["ref_mask"] = ref_mask

            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)

            # crop
            ref_img[ref_mask == 0, :] = 0

        elif False: #else:
            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)


        # from H,W,C to C,H,W
        #ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
        data_dict["ref_img"] = ref_img
        
        data_dict["scene_name"] = scene_name

        return data_dict




#    def __getitem_old__(self, idx):
#        """
#        data_dict:
#            "ref_img": (3, H, W)
#            "ref_pose": (3)
#            "src_img": (L, 3, H, W)
#            "src_pose": (L, 3)
#            "ref_mask": (H, W)
#            "src_mask": (L, H, W)
#            "ref_depth": (fW), ground truth depth of the reference img
#            "src_depth": (L, fW), ground truth depth of the source img
#        """
#        if self.start_scene is not None:
#            idx += self.scene_start_idx[self.start_scene]
#
#        # get the scene name according to the idx
#        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
#        scene_name = self.scene_names[scene_idx]
#
#        # get idx within scene
#        idx_within_scene = idx - self.scene_start_idx[scene_idx]
#
#        # get reference depth
#        ref_depth = self.gt_depth[scene_idx][idx_within_scene * (self.L + 1) + self.L]
#        data_dict = {"ref_depth": ref_depth}
#
#        # get source depth
#        src_depth = np.stack(
#            self.gt_depth[scene_idx][
#                idx_within_scene * (self.L + 1) : idx_within_scene * (self.L + 1)
#                + self.L
#            ],
#            axis=0,
#        )
#        data_dict["src_depth"] = src_depth
#
#        # get reference pose
#        ref_pose = self.gt_pose[scene_idx][idx_within_scene * (self.L + 1) + self.L]
#        data_dict["ref_noise"] = 0
#        data_dict["ref_pose"] = ref_pose
#
#        # get source pose
#        src_pose = np.stack(
#            self.gt_pose[scene_idx][
#                idx_within_scene * (self.L + 1) : idx_within_scene * (self.L + 1)
#                + self.L
#            ],
#            axis=0,
#        )
#        data_dict["src_noise"] = 0
#        data_dict["src_pose"] = src_pose
#
#        # get source images
#        src_img = []
#        for l in range(self.L):
#            image_path = os.path.join(
#                self.dataset_dir,
#                scene_name,
#                "rgb",
#                str(idx_within_scene).zfill(5) + "-" + str(l) + ".png",
#            )
#            src_img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
#            src_img.append(src_img_l)
#        src_img = np.stack(src_img, axis=0).astype(np.float32)  # (L, H, W, 3)
#        # get reference image
#        image_path = os.path.join(
#            self.dataset_dir,
#            scene_name,
#            "rgb",
#            str(idx_within_scene).zfill(5) + "-" + str(self.L) + ".png",
#        )
#        ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(
#            np.float32
#        )  # (H, W, 3)
#
#        if self.add_rp:
#            # reference
#            # get virtual roll and pitch
#            r = (np.random.random() - 0.5) * 2 * self.roll
#            p = (np.random.random() - 0.5) * 2 * self.pitch
#            # generate mask
#            mask = np.ones(list(ref_img.shape[:2]))
#            mask = gravity_align(mask, r, p, visualize=False, mode=1)
#            mask[mask < 1] = 0
#            ref_mask = mask.astype(np.uint8)
#            data_dict["ref_mask"] = ref_mask
#
#            # source
#            src_mask = []
#            for l in range(self.L):
#                # get virtual roll and pitch
#                r = (np.random.random() - 0.5) * 2 * self.roll
#                p = (np.random.random() - 0.5) * 2 * self.pitch
#                # generate mask
#                mask = np.ones(list(ref_img.shape[:2]))
#                mask = gravity_align(mask, r, p, visualize=False, mode=1)
#                mask[mask < 1] = 0
#                src_mask.append(mask.astype(np.uint8))
#            src_mask = np.stack(src_mask, axis=0)  # (L, H, W)
#            data_dict["src_mask"] = src_mask
#
#            # normalize => crop
#            # normailize
#            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
#            ref_img -= (0.485, 0.456, 0.406)
#            ref_img /= (0.229, 0.224, 0.225)
#
#            # crop
#            ref_img[ref_mask == 0, :] = 0
#
#            # source
#            for l in range(self.L):
#                # normalize => crop
#                # normailize
#                src_img[l, :, :, :] = (
#                    cv2.cvtColor(src_img[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
#                )
#                src_img[l, :, :, :] -= (0.485, 0.456, 0.406)
#                src_img[l, :, :, :] /= (0.229, 0.224, 0.225)
#
#                # crop
#                src_img[l, :, :, :][src_mask[l, :, :] == 0, :] = 0
#        else:
#            # normalize => crop
#            # normailize
#            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
#            ref_img -= (0.485, 0.456, 0.406)
#            ref_img /= (0.229, 0.224, 0.225)
#
#            # source
#            for l in range(self.L):
#                # normalize => crop
#                # normailize
#                src_img[l, :, :, :] = (
#                    cv2.cvtColor(src_img[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
#                )
#                src_img[l, :, :, :] -= (0.485, 0.456, 0.406)
#                src_img[l, :, :, :] /= (0.229, 0.224, 0.225)
#
#        # from H,W,C to C,H,W
#        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
#        src_img = np.transpose(src_img, (0, 3, 1, 2)).astype(np.float32)
#        data_dict["ref_img"] = ref_img
#        data_dict["src_img"] = src_img
#
#        return data_dict

class GridSeqDataset_HGE(Dataset):

    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        add_rp=False,
        roll=0,
        pitch=0,
        start_scene=None,
        end_scene=None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.start_scene = start_scene
        self.end_scene = end_scene
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []
        self.rgbs = []
        self.load_scene_start_idx_and_depths_and_poses()
        if start_scene is None:
            self.N = self.scene_start_idx[-1]
        else:
            # compute N
            self.N = (
                self.scene_start_idx[self.end_scene + 1]
                - self.scene_start_idx[self.start_scene]
            )

    def __len__(self):
        return self.N

    def load_scene_start_idx_and_depths_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        for scene in self.scene_names:

            # get depths
            if self.depth_dir == None:
                depth_file = os.path.join(
                    self.dataset_dir, scene, self.depth_suffix + ".txt"
                )
            else:
                depth_file = os.path.join(
                    self.depth_dir, scene, self.depth_suffix + ".txt"
                )

            # read depth
            with open(depth_file, "r") as f:
                depths_txt = [line.strip() for line in f.readlines()]

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            traj_len = len(poses_txt)
            scene_depths = []
            scene_poses = []
            for state_id in range(traj_len):
                # get depth
                depth = depths_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth]).astype(np.float32)
                scene_depths.append(depth)

                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)

            start_idx += traj_len // (self.L + 1)
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)

    def __getitem__(self, idx):
        """
        data_dict:
            "ref_img": (3, H, W)
            "ref_pose": (3)
            "src_img": (L, 3, H, W)
            "src_pose": (L, 3)
            "ref_mask": (H, W)
            "src_mask": (L, H, W)
            "ref_depth": (fW), ground truth depth of the reference img
            "src_depth": (L, fW), ground truth depth of the source img
        """
        if self.start_scene is not None:
            idx += self.scene_start_idx[self.start_scene]

        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # get reference depth
        ref_depth = self.gt_depth[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict = {"ref_depth": ref_depth}

        # # get source depth
        # src_depth = np.stack(
        #     self.gt_depth[scene_idx][
        #         idx_within_scene * (self.L + 1) : idx_within_scene * (self.L + 1)
        #         + self.L
        #     ],
        #     axis=0,
        # )
        # data_dict["src_depth"] = src_depth

        # get reference pose
        ref_pose = self.gt_pose[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict["ref_noise"] = 0
        data_dict["ref_pose"] = ref_pose

        # # get source pose
        # src_pose = np.stack(
        #     self.gt_pose[scene_idx][
        #         idx_within_scene * (self.L + 1) : idx_within_scene * (self.L + 1)
        #         + self.L
        #     ],
        #     axis=0,
        # )
        # data_dict["src_noise"] = 0
        # data_dict["src_pose"] = src_pose

        # # get source images
        # src_img = []
        # for l in range(self.L):
        #     image_path = os.path.join(
        #         self.dataset_dir,
        #         scene_name,
        #         "rgb",
        #         str(idx_within_scene).zfill(5) + "-" + str(l) + ".png",
        #     )
        #     src_img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
        #     src_img.append(src_img_l)
        # src_img = np.stack(src_img, axis=0).astype(np.float32)  # (L, H, W, 3)
        
        # get reference image

        # image_path = os.path.join(
        #     self.dataset_dir,
        #     scene_name,
        #     "rgb",
        #     str(idx_within_scene).zfill(5) + "-" + str(self.L) + ".png",
        # )


        directory_scene_rgb = os.path.join(self.dataset_dir, scene_name, "rgb")
        all_files = os.listdir(directory_scene_rgb)
        all_files.sort()

        image_path = os.path.join(
            directory_scene_rgb,
            all_files[idx_within_scene],
        )
        #print(image_path)

        #image_path = os.path.join(
        #    self.dataset_dir,
        #    scene_name,
        #    "rgb",
        #    str(idx_within_scene).zfill(5) + ".png",
        #)
        #print(image_path)

        ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(
            np.float32
        )  # (H, W, 3)

        if self.add_rp:
            # reference
            # get virtual roll and pitch
            r = (np.random.random() - 0.5) * 2 * self.roll
            p = (np.random.random() - 0.5) * 2 * self.pitch
            # generate mask
            mask = np.ones(list(ref_img.shape[:2]))
            #mask = gravity_align(mask, r, p, visualize=False, mode=1)
            mask = gravity_align(mask, r, p, mode=1)
            mask[mask < 1] = 0
            ref_mask = mask.astype(np.uint8)
            data_dict["ref_mask"] = ref_mask

            # # source
            # src_mask = []
            # for l in range(self.L):
            #     # get virtual roll and pitch
            #     r = (np.random.random() - 0.5) * 2 * self.roll
            #     p = (np.random.random() - 0.5) * 2 * self.pitch
            #     # generate mask
            #     mask = np.ones(list(ref_img.shape[:2]))
            #     #mask = gravity_align(mask, r, p, visualize=False, mode=1)
            #     mask = gravity_align(mask, r, p, mode=1)
            #     mask[mask < 1] = 0
            #     src_mask.append(mask.astype(np.uint8))
            # src_mask = np.stack(src_mask, axis=0)  # (L, H, W)
            # data_dict["src_mask"] = src_mask

            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)

            # crop
            ref_img[ref_mask == 0, :] = 0

            # # source
            # for l in range(self.L):
            #     # normalize => crop
            #     # normailize
            #     src_img[l, :, :, :] = (
            #         cv2.cvtColor(src_img[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
            #     )
            #     src_img[l, :, :, :] -= (0.485, 0.456, 0.406)
            #     src_img[l, :, :, :] /= (0.229, 0.224, 0.225)
# 
            #     # crop
            #     src_img[l, :, :, :][src_mask[l, :, :] == 0, :] = 0
        else:
            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)

            # # source
            # for l in range(self.L):
            #     # normalize => crop
            #     # normailize
            #     src_img[l, :, :, :] = (
            #         cv2.cvtColor(src_img[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
            #     )
            #     src_img[l, :, :, :] -= (0.485, 0.456, 0.406)
            #     src_img[l, :, :, :] /= (0.229, 0.224, 0.225)

        # from H,W,C to C,H,W
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
        # src_img = np.transpose(src_img, (0, 3, 1, 2)).astype(np.float32)
        data_dict["ref_img"] = ref_img
        # data_dict["src_img"] = src_img

        return data_dict