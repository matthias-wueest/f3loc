"""
Dataset for training structural depth prediction
"""

import os

import cv2
import numpy as np
import time
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from scipy.ndimage import zoom

from utils.utils import gravity_align, gravity_align_depth, gravity_align_normal, gravity_align_segmentation
#from metric3d.hubconf import *
from src.depth_image_functions import *

from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
import depthpro.src.depth_pro


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





class TrajDataset_gibson_metric3d_offline(Dataset):
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

        # load depth images
        depth_imgs = []
        for l in range(self.L):
            depth_path = os.path.join(
                self.dataset_dir, 
                scene_name, 
                "depth", 
                str(idx_within_scene * self.L + l).zfill(5) + "-0" + ".npy"
                )
            depth_img = np.load(depth_path).astype(np.float32)
            depth_imgs.append(depth_img)
            print("Depth image ", l, " done.")
        depth_imgs = np.stack(depth_imgs, axis=0).astype(np.float32)  # (L, H, W, 1)
        print(depth_imgs.shape)

        # prepare output
        data_dict["imgs"] = depth_imgs

        return data_dict




class TrajDataset_gibson_depthanything(Dataset):
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

        ####
        # DepthAnything Encoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        encoder =  'vitl' # or 'vitl', 'vitb', 'vits'
        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        self.da = DepthAnythingV2(**model_configs[encoder])
        self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
        self.da = self.da.to(device)  # Move the model to GPU
        self.da.eval()
        ####

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

        ## from H,W,C to C,H,W
        #imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)
        #data_dict["imgs"] = imgs

        # from H,W,C to C,H,W
        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)

        ref_imgs = []
        total_time = 0 # To store the total time taken for all iterations
        num_iterations = len(imgs)
        for ref_img in imgs:
            
            start_time = time.time()  # Record the start time of this iteration

            # depthanything encoder
            _, h, w = ref_img.shape
            #_, _, h, w = ref_img.shape
            #print("---------------------------")
            #print("ref_img.shape: ", ref_img.shape)
            input_tensor, _ =  self.da.image2tensor_simplified(ref_img)
            #print("input_tensor.shape: ", input_tensor.shape)
            with torch.no_grad():
                ref_img = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
            ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, 1024, fH, fW)
            ref_img = ref_img.squeeze(0) # (1024, fH, fW)
            ref_imgs.append(ref_img)

            # Calculate the time taken for this iteration
            iteration_time = time.time() - start_time
            total_time += iteration_time  # Accumulate total time

        # After the loop, calculate the average iteration time
        average_time = total_time / num_iterations
        print(f"Average iteration time: {average_time:.4f} seconds")

        ref_imgs = torch.stack(ref_imgs, dim=0)

        data_dict["imgs"] = ref_imgs        


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
        depth_imgs = np.stack(depth_imgs, axis=0).astype(np.float32)  # (L, H, W, 1)
        print(depth_imgs.shape)

        # gravity align depth images
        masks = []
        for l in range(self.L):
            mask = np.ones(list(depth_imgs.shape[1:3]))
            r=ref_euler_angles[l][1]
            p=-(ref_euler_angles[l][0]+np.pi/2)
            depth_imgs[l, :, :], mask = gravity_align_depth(depth_imgs[l, :, :], r, p, K=self.K)
            mask[mask < 1] = 0
            masks.append(mask.astype(np.uint8))
        masks = np.stack(masks, axis=0)  # (L, H, W)
        data_dict["masks"] = masks

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



class TrajDataset_hge_customized_metric3d_offline(Dataset):
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
        encoder = "metric3d_large",
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
        self.encoder = encoder
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


        # # get depth images
        # depth_imgs = []
        # for l in range(self.L):
        #     depth_img = get_depth_image_metric3d(imgs[l, :, :, :])#, K=self.K)
        #     depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array
        #     depth_imgs.append(depth_img)
        #     print("Depth image ", l, " done.")
        # depth_imgs = np.stack(depth_imgs, axis=0).astype(np.float32)  # (L, H, W, 1)
        # print(depth_imgs.shape)

        # load depth images
        depth_imgs = []
        for l in range(self.L):

            if self.encoder == "metric3d_large":
                depth_path = os.path.join(
                    self.dataset_dir, 
                    scene_name, 
                    "depth", 
                    str(idx_within_scene * self.L + l).zfill(5) + "-0" + ".npy"
                    )
            elif self.encoder == "metric3d_small":
                depth_path = os.path.join(
                    self.dataset_dir, 
                    scene_name, 
                    "depth_small", 
                    str(idx_within_scene * self.L + l).zfill(5) + "-0" + ".npy"
                    )                 
            depth_img = np.load(depth_path).astype(np.float32)
            depth_imgs.append(depth_img)
            print("Depth image ", l, " done.")
        depth_imgs = np.stack(depth_imgs, axis=0).astype(np.float32)  # (L, H, W, 1)
        print(depth_imgs.shape)


        # gravity align depth images
        masks = []
        for l in range(self.L):
            mask = np.ones(list(depth_imgs.shape[1:3]))
            r=ref_euler_angles[l][1]
            p=-(ref_euler_angles[l][0]+np.pi/2)
            depth_imgs[l, :, :], mask = gravity_align_depth(depth_imgs[l, :, :], r, p, K=self.K)
            mask[mask < 1] = 0
            masks.append(mask.astype(np.uint8))
        masks = np.stack(masks, axis=0)  # (L, H, W)
        data_dict["masks"] = masks

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




class TrajDataset_hge_customized_metric3d_removed_horizontals(Dataset):
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


        depth_imgs = []
        masks = []
        for l in range(self.L):

            # get depth images and normal images
            depth_img, normal_img = get_depth_image_metric3d_normals(imgs[l, :, :, :], K=self.K)
            depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array
            normal_img = normal_img.cpu().numpy().transpose((1, 2, 0))  # Move to CPU and convert to numpy array

            # gravity align depth images
            r=ref_euler_angles[l][1]
            p=-(ref_euler_angles[l][0]+np.pi/2)
            depth_img_aligned, depth_mask = gravity_align_depth(depth_img, r, p, K=self.K)
            
            # gravity align normal images
            #print(normal_img.shape)
            #print(depth_img.shape)
            normal_img_aligned = gravity_align_normal(normal_img, depth_img, r, p, K=self.K)
            
            # extract normal mask
            threshold_angle_cos = 0.9  # Cosine of threshold angle (e.g., cos(18°) ≈ 0.95)

            vertical_vector_ground = np.array([0, -1, 0])  # Assuming y-axis is vertical
            dot_products_ground = np.dot(normal_img_aligned, vertical_vector_ground)
            horizontal_planes_ground_mask = dot_products_ground > threshold_angle_cos

            vertical_vector_ceiling = np.array([0, 1, 0])  # Assuming y-axis is vertical
            dot_products_ceiling = np.dot(normal_img_aligned, vertical_vector_ceiling)
            horizontal_planes_ceiling_mask = dot_products_ceiling > threshold_angle_cos

            normal_mask = (horizontal_planes_ground_mask | horizontal_planes_ceiling_mask).astype(np.uint8)
            normal_mask = 1 - normal_mask

            # combine masks
            mask = (depth_mask & normal_mask).astype(np.uint8)

            # apply combined mask
            depth_img_aligned_masked = depth_img_aligned
            depth_img_aligned_masked[mask<1] = 0
            
            # store results
            depth_imgs.append(depth_img_aligned_masked)
            masks.append(mask)

            print("Depth image ", l, " done.")

        depth_imgs = np.stack(depth_imgs, axis=0).astype(np.float32)  # (L, H, W, 1)
        masks = np.stack(masks, axis=0).astype(np.uint8)  # (L, H, W)
        print(depth_imgs.shape)
        print(masks.shape)
                    
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
        data_dict["masks"] = masks

        return data_dict


class TrajDataset_hge_customized_metric3d_depths_normals(Dataset):
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

        depth_imgs = []
        normal_imgs = []
        masks = []
        for l in range(self.L):

            # get depth images and normal images
            depth_img, normal_img = get_depth_image_metric3d_normals(imgs[l, :, :, :], K=self.K)
            depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array
            normal_img = normal_img.cpu().numpy().transpose((1, 2, 0))  # Move to CPU and convert to numpy array

            # gravity align depth images
            r=ref_euler_angles[l][1]
            p=-(ref_euler_angles[l][0]+np.pi/2)
            depth_img_aligned, depth_mask = gravity_align_depth(depth_img, r, p, K=self.K)
            
            # gravity align normal images
            #print(normal_img.shape)
            #print(depth_img.shape)
            normal_img_aligned = gravity_align_normal(normal_img, depth_img, r, p, K=self.K)
            
            # store results
            depth_imgs.append(depth_img_aligned)
            normal_imgs.append(normal_img_aligned)
            masks.append(depth_mask)

        depth_imgs = np.stack(depth_imgs, axis=0).astype(np.float32)  # (L, H, W, 1)
        normal_imgs = np.stack(normal_imgs, axis=0).astype(np.float32)  # (L, H, W, 3)
        masks = np.stack(masks, axis=0).astype(np.uint8)  # (L, H, W)
        #print(depth_imgs.shape)
        #print(normal_imgs.shape)
        #print(masks.shape)

        data_dict["imgs"] = np.concatenate((depth_imgs, normal_imgs), axis=-1)
        data_dict["masks"] = masks

        return data_dict        


        # get depth images
        depth_imgs = []
        for l in range(self.L):
            depth_img = get_depth_image_metric3d(imgs[l, :, :, :])#, K=self.K)
            depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array
            depth_imgs.append(depth_img)
            print("Depth image ", l, " done.")
        depth_imgs = np.stack(depth_imgs, axis=0).astype(np.float32)  # (L, H, W, 1)
        #print(depth_imgs.shape)

        # gravity align depth images
        masks = []
        for l in range(self.L):
            mask = np.ones(list(depth_imgs.shape[1:3]))
            r=ref_euler_angles[l][1]
            p=-(ref_euler_angles[l][0]+np.pi/2)
            depth_imgs[l, :, :], mask = gravity_align_depth(depth_imgs[l, :, :], r, p, K=self.K)
            mask[mask < 1] = 0
            masks.append(mask.astype(np.uint8))
        masks = np.stack(masks, axis=0)  # (L, H, W)
        data_dict["masks"] = masks

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



class TrajDataset_hge_customized_cropped_gravity_align_dinov2(Dataset):
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

#        ####
#        # DepthAnything Encoder
#        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#        print(f"Using device: {device}")
#        
#        model_configs = {
#            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
#        }
#        encoder = 'vits' #'vitb' #'vits' # or 'vitl', 
#        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
#        self.da = DepthAnythingV2(**model_configs[encoder])
#        self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
#        self.da = self.da.to(device)  # Move the model to GPU
#        self.da.eval()
#        ####

        ####
        # DinoV2 Encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BACKBONE_SIZE = "large" # in ("small", "base", "large" or "giant")
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"
        self.backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        self.backbone_model = self.backbone_model.to(self.device)
        self.backbone_model.eval()
        ###

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


        # from H,W,C to C,H,W
        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)

        ref_imgs = []
        total_time = 0 # To store the total time taken for all iterations
        num_iterations = len(imgs)
        for ref_img in imgs:

            start_time = time.time()  # Record the start time of this iteration

#            ###
#            # depthanything encoder
#            _, h, w = ref_img.shape
#            #_, _, h, w = ref_img.shape
#            #print("---------------------------")
#            #print("ref_img.shape: ", ref_img.shape)
#            input_tensor, _ =  self.da.image2tensor_simplified(ref_img)
#            #print("input_tensor.shape: ", input_tensor.shape)
#            with torch.no_grad():
#                ref_img = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
#            ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, 1024, fH, fW)
#            ref_img = ref_img.squeeze(0) # (1024, fH, fW)
#            ref_imgs.append(ref_img)
#            ###

            ###
            # DinoV2 encoder
            _, h, w = ref_img.shape # C,H,W

            # Compute scale factors for resizing to (224, 224)
            scale_factors = (1, 224 / h, 224 / w)  # (Channels stay same, scale height, scale width)
            resized_img = zoom(ref_img, scale_factors, order=1)  # order=1 means bilinear interpolation
            input_tensor = torch.from_numpy(resized_img).unsqueeze(0)
            #print("input_tensor.shape", input_tensor.shape)
            input_tensor = input_tensor.to(self.device)

            # Perform inference
            with torch.no_grad():  # Disable gradient computation for inference
                ref_img  = self.backbone_model.forward_features(input_tensor)['x_norm_patchtokens']
            #print("ref_img.shape: ", ref_img.shape)
            batch_size, num_patches, embedding_size = ref_img.shape     #torch.Size([B, 256, 384])
            fH = fW = int(num_patches ** 0.5)  # Calculate spatial dimensions (16x16 grid for 256 patches)
            ref_img = ref_img.view(1, fH, fW, embedding_size).permute(0, 3, 1, 2) # Shape: [1, embedding_size, 16, 16]
            #ref_img = ref_img.view(batch_size, embedding_size, fH, fW)  # Shape: [1, embedding_size, 16, 16]
            #print("ref_img.shape 2: ", ref_img.shape)
            # Interpolate to get desired shape
            ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, C, fH, fW)
            ref_img = ref_img.squeeze(0) # (C, fH, fW)
            # Bugfix
            #print("bugfix ref_img.shape: ", ref_img.shape)
            ref_imgs.append(ref_img)
            ###



            # Calculate the time taken for this iteration
            iteration_time = time.time() - start_time
            total_time += iteration_time  # Accumulate total time

        # After the loop, calculate the average iteration time
        average_time = total_time / num_iterations
        print(f"Average iteration time: {average_time:.4f} seconds")


        ref_imgs = torch.stack(ref_imgs, dim=0)

        data_dict["imgs"] = ref_imgs

        return data_dict


class TrajDataset_hge_customized_cropped_gravity_align_depthpro(Dataset):
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

#        ####
#        # DepthAnything Encoder
#        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#        print(f"Using device: {device}")
#        
#        model_configs = {
#            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
#        }
#        encoder = 'vits' #'vitb' #'vits' # or 'vitl', 
#        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
#        self.da = DepthAnythingV2(**model_configs[encoder])
#        self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
#        self.da = self.da.to(device)  # Move the model to GPU
#        self.da.eval()
#        ####

        ####
        # DepthPro Encoder
        model, transform = depthpro.src.depth_pro.create_model_and_transforms()
        model.eval()
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        self.model = model
        self.transform = transform
        self.device = device
        ####

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
        image_path_ls = []
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
            image_path_ls.append(image_path)
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
        original_aligned_imgs = imgs.copy()

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
            original_aligned_imgs[l, :, :, :][masks[l, :, :] == 0, :] = 0
        
###


        # from H,W,C to C,H,W
        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)
        original_aligned_imgs = np.transpose(original_aligned_imgs, (0, 3, 1, 2)).astype(np.float32)

        ref_imgs = []
        combined_features_ls = []
        total_time = 0 # To store the total time taken for all iterations
        num_iterations = len(imgs)
        for l, original_aligned_img in enumerate(original_aligned_imgs):

            start_time = time.time()  # Record the start time of this iteration

#            # depthanything encoder
#            _, h, w = ref_img.shape
#            #_, _, h, w = ref_img.shape
#            #print("---------------------------")
#            #print("ref_img.shape: ", ref_img.shape)
#            input_tensor, _ =  self.da.image2tensor_simplified(ref_img)
#            #print("input_tensor.shape: ", input_tensor.shape)
#            with torch.no_grad():
#                ref_img = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
#            ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, 1024, fH, fW)
#            ref_img = ref_img.squeeze(0) # (1024, fH, fW)
#            ref_imgs.append(ref_img)

            ####
            ## depthpro encoder
            _, h, w = original_aligned_img.shape

            # Load and preprocess an image.
            image, _, f_px = depthpro.src.depth_pro.load_rgb(image_path_ls[l])
            image = self.transform(image)        
            image = image.to(self.device)
            encodings = self.model.infer2(image)

            resized_encodings = []
            for encoding in encodings:
                # Resize to (1, C, target_height, target_width) using bilinear interpolation
                resized_encoding = F.interpolate(encoding, size=(h//16, w//16), mode='bilinear', align_corners=False)
                resized_encodings.append(resized_encoding)

            # Concatenate along the channel dimension, resulting in a shape of (1, sum(C_i), target_height, target_width)
            combined_features = torch.cat(resized_encodings, dim=1)        
            combined_features = combined_features.squeeze(0)
            combined_features_ls.append(combined_features)
            #print("combined_features.shape: ", combined_features.shape)
            ####


            # Calculate the time taken for this iteration
            iteration_time = time.time() - start_time
            total_time += iteration_time  # Accumulate total time

        # After the loop, calculate the average iteration time
        average_time = total_time / num_iterations
        print(f"Average iteration time: {average_time:.4f} seconds")

        #ref_imgs = torch.stack(ref_imgs, dim=0)
        #data_dict["imgs"] = ref_imgs

        combined_features = torch.stack(combined_features_ls, dim=0)
        data_dict["imgs"] = combined_features

        return data_dict



class TrajDataset_hge_customized_cropped_gravity_align_depthanything(Dataset):
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

        ####
        # DepthAnything Encoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        encoder = 'vits' #'vitb' #'vits' # or 'vitl', 
        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        self.da = DepthAnythingV2(**model_configs[encoder])
        self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
        self.da = self.da.to(device)  # Move the model to GPU
        self.da.eval()
        ####


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


        # from H,W,C to C,H,W
        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)

        ref_imgs = []
        total_time = 0 # To store the total time taken for all iterations
        num_iterations = len(imgs)
        for ref_img in imgs:

            start_time = time.time()  # Record the start time of this iteration

            # depthanything encoder
            _, h, w = ref_img.shape
            #_, _, h, w = ref_img.shape
            #print("---------------------------")
            #print("ref_img.shape: ", ref_img.shape)
            input_tensor, _ =  self.da.image2tensor_simplified(ref_img)
            #print("input_tensor.shape: ", input_tensor.shape)
            with torch.no_grad():
                ref_img = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
            ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, 1024, fH, fW)
            ref_img = ref_img.squeeze(0) # (1024, fH, fW)
            ref_imgs.append(ref_img)

            # Calculate the time taken for this iteration
            iteration_time = time.time() - start_time
            total_time += iteration_time  # Accumulate total time

        # After the loop, calculate the average iteration time
        average_time = total_time / num_iterations
        print(f"Average iteration time: {average_time:.4f} seconds")


        ref_imgs = torch.stack(ref_imgs, dim=0)

        data_dict["imgs"] = ref_imgs

        return data_dict



class TrajDataset_hge_customized_cropped_gravity_align_depthanything_semantics(Dataset):
    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        semantic_dir=None,
        semantic_suffix="semantic40",
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
        self.semantic_dir = semantic_dir
        self.semantic_suffix = semantic_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_semantic = []
        self.gt_pose = []
        self.rgbs = []
        self.without_depth = without_depth
        self.K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])
        self.gt_euler_angles = []
        self.load_scene_start_idx_and_depths_and_poses()
        self.N = self.scene_start_idx[-1]

        ####
        # DepthAnything Encoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        encoder =  'vitl' # or 'vitl', 'vitb', 'vits'
        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        self.da = DepthAnythingV2(**model_configs[encoder])
        self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
        self.da = self.da.to(device)  # Move the model to GPU
        self.da.eval()
        ####


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


                # get semantics
                if self.semantic_dir == None:
                    semantic_file = os.path.join(
                        self.dataset_dir, scene, self.semantic_suffix + ".txt"
                    )
                else:
                    semantic_file = os.path.join(
                        self.semantic_dir, scene, self.semantic_suffix + ".txt"
                    )
                # read semantic
                with open(semantic_file, "r") as f:
                    semantics_txt = [line.strip() for line in f.readlines()]

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
            scene_semantics = []
            scene_poses = []
            scene_euler_angles = []
            for state_id in range(traj_len):
                if not self.without_depth:
                    # get depth
                    depth = depths_txt[state_id].split(" ")
                    depth = np.array([float(d) for d in depth]).astype(np.float32)
                    scene_depths.append(depth)

                    # get semantic
                    semantic = semantics_txt[state_id].split(" ")
                    semantic = np.array([float(s) for s in semantic]).astype(np.float32)
                    scene_semantics.append(semantic)

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
            self.gt_semantic.append(scene_semantics)
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

            # get semantics
            gt_semantic = self.gt_semantic[scene_idx][idx_within_scene * self.L : idx_within_scene * self.L + self.L]
            gt_semantic = torch.tensor(gt_semantic)
            gt_semantic = gt_semantic.long()  # Convert to long (integer) type
            gt_semantic = gt_semantic - 1     # Convert to 0-indexed labels
            data_dict["gt_semantic"] = gt_semantic
        
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
        
        imgs_rgb = imgs.copy()
        for l in range(self.L):
            imgs_rgb[l, :, :, :] = cv2.cvtColor(imgs_rgb[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
        data_dict["imgs_rgb"] = imgs_rgb

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


        # from H,W,C to C,H,W
        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)

        ref_imgs = []
        for ref_img in imgs:
            # depthanything encoder
            _, h, w = ref_img.shape
            #_, _, h, w = ref_img.shape
            #print("---------------------------")
            #print("ref_img.shape: ", ref_img.shape)
            input_tensor, _ =  self.da.image2tensor_simplified(ref_img)
            #print("input_tensor.shape: ", input_tensor.shape)
            with torch.no_grad():
                ref_img = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
            ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, 1024, fH, fW)
            ref_img = ref_img.squeeze(0) # (1024, fH, fW)
            ref_imgs.append(ref_img)
        ref_imgs = torch.stack(ref_imgs, dim=0)

        data_dict["imgs"] = ref_imgs

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


class GridSeqDataset_hge_customized_metric3d(Dataset):

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


        # get depth image
        depth_img = get_depth_image_metric3d(ref_img)
        depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array
        #print("Depth image done.")

        # gravity align depth image
        r=ref_euler_angles[1]
        p=-(ref_euler_angles[0]+np.pi/2)
        depth_img, depth_mask = gravity_align_depth(depth_img, r, p, K=self.K)

        # # generate mask
        # mask = np.ones(list(ref_img.shape[:2]))
        # mask = gravity_align(mask, r, p, mode=1)
        # mask[mask < 1] = 0
        # ref_mask = mask.astype(np.uint8)
        
        # results dict
        #print("depth_img.shape: ", depth_img.shape)
        #print("ref_mask.shape: ", ref_mask.shape)

        data_dict["ref_img"] = depth_img
        data_dict["ref_mask"] = depth_mask
        data_dict["scene_name"] = scene_name

        return data_dict        


class GridSeqDataset_hge_customized_metric3d_offline(Dataset):

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
        encoder = "metric3d_large"
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
        self.encoder = encoder
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

        # # get reference image
        # image_path = os.path.join(
        #     self.dataset_dir,
        #     scene_name,
        #     "rgb",
        #     #str(idx_within_scene).zfill(5) + "-" + str(self.L) + ".png",
        #     str(idx_within_scene).zfill(5) + "-0" + ".jpg",
        # )
        # ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(
        #     np.float32
        # )  # (H, W, 3)

        ## get depth image
        #depth_img = get_depth_image_metric3d(ref_img)
        #depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array
        ##print("Depth image done.")

        # load depth image
        if self.encoder == "metric3d_large":
            depth_path = os.path.join(
                self.dataset_dir, 
                scene_name, 
                "depth", 
                str(idx_within_scene).zfill(5) + "-0" + ".npy"
                )
        elif self.encoder == "metric3d_small":
            depth_path = os.path.join(
                self.dataset_dir, 
                scene_name, 
                "depth_small", 
                str(idx_within_scene).zfill(5) + "-0" + ".npy"
                )
        depth_img = np.load(depth_path).astype(np.float32)
        #print("Depth image done.")

        # gravity align depth image
        r=ref_euler_angles[1]
        p=-(ref_euler_angles[0]+np.pi/2)
        depth_img, depth_mask = gravity_align_depth(depth_img, r, p, K=self.K)
        #print("Gravity alignment done.")

        # # generate mask
        # mask = np.ones(list(ref_img.shape[:2]))
        # mask = gravity_align(mask, r, p, mode=1)
        # mask[mask < 1] = 0
        # ref_mask = mask.astype(np.uint8)
        
        # results dict
        #print("depth_img.shape: ", depth_img.shape)
        #print("ref_mask.shape: ", ref_mask.shape)

        data_dict["ref_img"] = depth_img
        data_dict["ref_mask"] = depth_mask
        data_dict["scene_name"] = scene_name

        return data_dict  


class GridSeqDataset_hge_customized_metric3d_depths_normals(Dataset):

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


        # get depth image
        depth_img = get_depth_image_metric3d(ref_img)
        depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array

        # get depth image and normal image 
        depth_img, normal_img = get_depth_image_metric3d_normals(ref_img, K=self.K)
        depth_img = depth_img.cpu().numpy()
        normal_img = normal_img.cpu().numpy().transpose((1, 2, 0))

        # gravity align depth image
        r=ref_euler_angles[1]
        p=-(ref_euler_angles[0]+np.pi/2)
        depth_img, depth_mask = gravity_align_depth(depth_img, r, p, K=self.K)
        
        # gravity align normal image
        normal_img = gravity_align_normal(normal_img, depth_img, r, p, K=self.K).transpose((2, 0, 1))

        # # generate mask
        # mask = np.ones(list(ref_img.shape[:2]))
        # mask = gravity_align(mask, r, p, mode=1)
        # mask[mask < 1] = 0
        # ref_mask = mask.astype(np.uint8)
        
        # results dict
        #print("depth_img.shape: ", depth_img.shape)
        #print("ref_mask.shape: ", ref_mask.shape)

        data_dict["ref_img"] = np.vstack((depth_img[np.newaxis, ...], normal_img))
        data_dict["ref_mask"] = depth_mask
        data_dict["scene_name"] = scene_name

        return data_dict 



class GridSeqDataset_hge_customized_metric3d_depths_normals_segmentation(Dataset):

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


        # get depth image
        depth_img = get_depth_image_metric3d(ref_img)
        depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array

        # get depth image and normal image 
        depth_img, normal_img = get_depth_image_metric3d_normals(ref_img, K=self.K)
        depth_img = depth_img.cpu().numpy()
        normal_img = normal_img.cpu().numpy().transpose((1, 2, 0))

        # get segmentation image
        #print("ref_img.shape: ", ref_img.shape)
        segmentation_img = get_segmentation_image(ref_img, K=self.K)
        #print("segmentation_img before gravity: ", segmentation_img.shape)
        # gravity align depth image
        r=ref_euler_angles[1]
        p=-(ref_euler_angles[0]+np.pi/2)
        depth_img, depth_mask = gravity_align_depth(depth_img, r, p, K=self.K)
        
        # gravity align normal image
        normal_img = gravity_align_normal(normal_img, depth_img, r, p, K=self.K).transpose((2, 0, 1))

        # gravity align segmentation image
        segmentation_img = gravity_align_segmentation(segmentation_img, r, p, K=self.K)
        
        #print("depth_mask.shape: ", depth_mask.shape)
        #print("segmentation_img after gravity: ", segmentation_img.shape)


        segmentation_img[:, depth_mask<1] = -float('inf')
        segmentation_img = softmax_pytorch(segmentation_img) # logits => probs
        segmentation_img[:, depth_mask<1] = 0

        # # generate mask
        # mask = np.ones(list(ref_img.shape[:2]))
        # mask = gravity_align(mask, r, p, mode=1)
        # mask[mask < 1] = 0
        # ref_mask = mask.astype(np.uint8)
        
        # results dict
        #print("depth_img.shape: ", depth_img.shape)
        #print("ref_mask.shape: ", ref_mask.shape)

       #print("Depth image shape with new axis:", depth_img[np.newaxis, ...].shape)
       #print("Normal image shape:", normal_img.shape)
       #print("Segmentation image shape:", segmentation_img.shape)

        data_dict["ref_img"] = np.vstack((depth_img[np.newaxis, ...], normal_img, segmentation_img))
        data_dict["ref_mask"] = depth_mask
        data_dict["scene_name"] = scene_name

        return data_dict 





class GridSeqDataset_hge_customized_cropped_gravity_align_depthanything(Dataset):

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
        
        ####
        # DepthAnything Encoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        encoder =  'vitl' # 'vitl' # or 'vitl', 'vitb', 'vits'
        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        self.da = DepthAnythingV2(**model_configs[encoder])
        self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
        self.da = self.da.to(device)  # Move the model to GPU
        self.da.eval()
        ####  

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
        
        ###
        # depthanything encoder
        _, h, w = ref_img.shape
        #_, _, h, w = ref_img.shape
        #print("---------------------------")
        #print("ref_img.shape: ", ref_img.shape)
        input_tensor, _ =  self.da.image2tensor_simplified(ref_img)
        #print("input_tensor.shape: ", input_tensor.shape)
        with torch.no_grad():
            ref_img = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
        ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, 1024, fH, fW)
        ref_img = ref_img.squeeze(0) # (1024, fH, fW)
        #print("ref_img.shape: ", ref_img.shape)
        ###

        # store to dict
        data_dict["ref_img"] = ref_img
        data_dict["scene_name"] = scene_name

        return data_dict


class GridSeqDataset_hge_customized_cropped_gravity_align_depthpro(Dataset):

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
        
#        ####
#        # DepthAnything Encoder
#        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#        print(f"Using device: {device}")
#        
#        model_configs = {
#            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
#        }
#        encoder =  'vitb' # 'vitl' # or 'vitl', 'vitb', 'vits'
#        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
#        self.da = DepthAnythingV2(**model_configs[encoder])
#        self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
#        self.da = self.da.to(device)  # Move the model to GPU
#        self.da.eval()
#        ####

        ####
        # DepthPro Encoder
        model, transform = depthpro.src.depth_pro.create_model_and_transforms()
        model.eval()
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        self.model = model
        self.transform = transform
        self.device = device
        ####


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
        original_aligned_img = ref_img.copy()

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
        original_aligned_img[ref_mask == 0, :] = 0

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
        original_aligned_img = np.transpose(original_aligned_img, (2, 0, 1)).astype(np.float32)

#        # depthanything encoder
#        _, h, w = ref_img.shape
#        #_, _, h, w = ref_img.shape
#        #print("---------------------------")
#        #print("ref_img.shape: ", ref_img.shape)
#        input_tensor, _ =  self.da.image2tensor_simplified(ref_img)
#        #print("input_tensor.shape: ", input_tensor.shape)
#        with torch.no_grad():
#            ref_img = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
#        ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, 1024, fH, fW)
#        ref_img = ref_img.squeeze(0) # (1024, fH, fW)
#        #print("ref_img.shape: ", ref_img.shape)

        ####
        ## depthpro encoder
        _, h, w = original_aligned_img.shape

        # Load and preprocess an image.
        image, _, f_px = depthpro.src.depth_pro.load_rgb(image_path)
        image = self.transform(image)        
        image = image.to(self.device)
        #encodings = self.model.infer2(image)
        encodings = self.model.infer2(image, f_px=self.K[0,0])

        resized_encodings = []
        for encoding in encodings:
            # Resize to (1, C, target_height, target_width) using bilinear interpolation
            resized_encoding = F.interpolate(encoding, size=(h//16, w//16), mode='bilinear', align_corners=False)
            resized_encodings.append(resized_encoding)
        
        # Concatenate along the channel dimension, resulting in a shape of (1, sum(C_i), target_height, target_width)
        combined_features = torch.cat(resized_encodings, dim=1)        
        combined_features = combined_features.squeeze(0)
        #print("combined_features.shape: ", combined_features.shape)
        ####

        # store to dict
        data_dict["ref_img"] = combined_features
        #data_dict["ref_img"] = ref_img
        data_dict["scene_name"] = scene_name

        return data_dict



class GridSeqDataset_hge_customized_cropped_gravity_align_dinov2(Dataset):

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
        
#        ####
#        # DepthAnything Encoder
#        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#        print(f"Using device: {device}")
#        
#        model_configs = {
#            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
#        }
#        encoder =  'vitl' # 'vitl' # or 'vitl', 'vitb', 'vits'
#        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
#        self.da = DepthAnythingV2(**model_configs[encoder])
#        self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
#        self.da = self.da.to(device)  # Move the model to GPU
#        self.da.eval()
#        ####

        ####
        # DinoV2 Encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BACKBONE_SIZE = "large" # in ("small", "base", "large" or "giant")
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"
        self.backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        self.backbone_model = self.backbone_model.to(self.device)
        self.backbone_model.eval()
        ###      

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
        
#        ###
#        # depthanything encoder
#        _, h, w = ref_img.shape
#        #_, _, h, w = ref_img.shape
#        #print("---------------------------")
#        #print("ref_img.shape: ", ref_img.shape)
#        input_tensor, _ =  self.da.image2tensor_simplified(ref_img)
#        #print("input_tensor.shape: ", input_tensor.shape)
#        with torch.no_grad():
#            ref_img = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
#        ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, 1024, fH, fW)
#        ref_img = ref_img.squeeze(0) # (1024, fH, fW)
#        #print("ref_img.shape: ", ref_img.shape)
#        ###

        ###
        # DinoV2 encoder
        _, h, w = ref_img.shape # C,H,W

        # Compute scale factors for resizing to (224, 224)
        scale_factors = (1, 224 / h, 224 / w)  # (Channels stay same, scale height, scale width)
        resized_img = zoom(ref_img, scale_factors, order=1)  # order=1 means bilinear interpolation
        input_tensor = torch.from_numpy(resized_img).unsqueeze(0)
        #print("input_tensor.shape", input_tensor.shape)
        input_tensor = input_tensor.to(self.device)

        # Perform inference
        with torch.no_grad():  # Disable gradient computation for inference
            ref_img  = self.backbone_model.forward_features(input_tensor)['x_norm_patchtokens']
        #print("ref_img.shape: ", ref_img.shape)
        batch_size, num_patches, embedding_size = ref_img.shape     #torch.Size([B, 256, 384])
        fH = fW = int(num_patches ** 0.5)  # Calculate spatial dimensions (16x16 grid for 256 patches)
        ref_img = ref_img.view(1, fH, fW, embedding_size).permute(0, 3, 1, 2) # Shape: [1, embedding_size, 16, 16]
        #ref_img = ref_img.view(batch_size, embedding_size, fH, fW)  # Shape: [1, embedding_size, 16, 16]
        #print("ref_img.shape 2: ", ref_img.shape)
        # Interpolate to get desired shape
        ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, C, fH, fW)
        ref_img = ref_img.squeeze(0) # (C, fH, fW)
        # Bugfix
        #print("bugfix ref_img.shape: ", ref_img.shape)
        ###


        # store to dict
        data_dict["ref_img"] = ref_img
        data_dict["scene_name"] = scene_name

        return data_dict




class GridSeqDataset_hge_customized_cropped_gravity_align_depthanything_semantics(Dataset):

    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        depth_dir=None,
        depth_suffix="depth40",
        semantic_dir=None,
        semantic_suffix="semantic40",
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
        self.semantic_dir = semantic_dir
        self.semantic_suffix = semantic_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_semantic = []
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
        
        ####
        # DepthAnything Encoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        encoder =  'vitl' # or 'vitl', 'vitb', 'vits'
        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        self.da = DepthAnythingV2(**model_configs[encoder])
        self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
        self.da = self.da.to(device)  # Move the model to GPU
        self.da.eval()
        ####



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


            # get semantics
            if self.semantic_dir == None:
                semantic_file = os.path.join(
                    self.dataset_dir, scene, self.semantic_suffix + ".txt"
                )
            else:
                semantic_file = os.path.join(
                    self.semantic_dir, scene, self.semantic_suffix + ".txt"
                )
            # read semantic
            with open(semantic_file, "r") as f:
                semantics_txt = [line.strip() for line in f.readlines()]



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
            scene_semantics = []
            scene_poses = []
            scene_euler_angles = []
            for state_id in range(traj_len):
                # get depth
                depth = depths_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth]).astype(np.float32)
                scene_depths.append(depth)

                # get semantic
                semantic = semantics_txt[state_id].split(" ")
                semantic = np.array([float(s) for s in semantic]).astype(np.float32)
                scene_semantics.append(semantic)

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
            self.gt_semantic.append(scene_semantics)
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

        # get reference semantic
        ref_semantic = self.gt_semantic[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        ref_semantic = torch.tensor(ref_semantic)
        ref_semantic = ref_semantic.long()  # Convert to long (integer) type
        ref_semantic = ref_semantic - 1     # Convert to 0-indexed labels
        data_dict["ref_semantic"] = ref_semantic
        #print("ref_semantic.shape: ", ref_semantic.shape)

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


        # depthanything encoder
        _, h, w = ref_img.shape
        #_, _, h, w = ref_img.shape
        #print("---------------------------")
        #print("ref_img.shape: ", ref_img.shape)
        input_tensor, _ =  self.da.image2tensor_simplified(ref_img)
        #print("input_tensor.shape: ", input_tensor.shape)
        with torch.no_grad():
            ref_img = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
        ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, 1024, fH, fW)
        ref_img = ref_img.squeeze(0) # (1024, fH, fW)
        #print("ref_img.shape: ", ref_img.shape)

        # store to dict
        data_dict["ref_img"] = ref_img
        data_dict["scene_name"] = scene_name

        return data_dict







class GridSeqDataset_Gibson_depthanything(Dataset):

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

        ####
        # DepthAnything Encoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        encoder =  'vits' # or 'vitl', 'vitb', 'vits'
        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        self.da = DepthAnythingV2(**model_configs[encoder])
        self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
        self.da = self.da.to(device)  # Move the model to GPU
        self.da.eval()
        ####



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

        ## get source depth
        #src_depth = np.stack(
        #    self.gt_depth[scene_idx][
        #        idx_within_scene * (self.L + 1) : idx_within_scene * (self.L + 1)
        #        + self.L
        #    ],
        #    axis=0,
        #)
        #data_dict["src_depth"] = src_depth

        # get reference pose
        ref_pose = self.gt_pose[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict["ref_noise"] = 0
        data_dict["ref_pose"] = ref_pose

        ## get source pose
        #src_pose = np.stack(
        #    self.gt_pose[scene_idx][
        #        idx_within_scene * (self.L + 1) : idx_within_scene * (self.L + 1)
        #        + self.L
        #    ],
        #    axis=0,
        #)
        #data_dict["src_noise"] = 0
        #data_dict["src_pose"] = src_pose

        ## get source images
        #src_img = []
        #for l in range(self.L):
        #    image_path = os.path.join(
        #        self.dataset_dir,
        #        scene_name,
        #        "rgb",
        #        str(idx_within_scene).zfill(5) + "-" + str(l) + ".png",
        #    )
        #    src_img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
        #    src_img.append(src_img_l)
        #src_img = np.stack(src_img, axis=0).astype(np.float32)  # (L, H, W, 3)

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

            ## source
            #src_mask = []
            #for l in range(self.L):
            #    # get virtual roll and pitch
            #    r = (np.random.random() - 0.5) * 2 * self.roll
            #    p = (np.random.random() - 0.5) * 2 * self.pitch
            #    # generate mask
            #    mask = np.ones(list(ref_img.shape[:2]))
            #    #mask = gravity_align(mask, r, p, visualize=False, mode=1)
            #    mask = gravity_align(mask, r, p, mode=1)
            #    mask[mask < 1] = 0
            #    src_mask.append(mask.astype(np.uint8))
            #src_mask = np.stack(src_mask, axis=0)  # (L, H, W)
            #data_dict["src_mask"] = src_mask

            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)

            # crop
            ref_img[ref_mask == 0, :] = 0

            ## source
            #for l in range(self.L):
            #    # normalize => crop
            #    # normailize
            #    src_img[l, :, :, :] = (
            #        cv2.cvtColor(src_img[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
            #    )
            #    src_img[l, :, :, :] -= (0.485, 0.456, 0.406)
            #    src_img[l, :, :, :] /= (0.229, 0.224, 0.225)
            #
            #    # crop
            #    src_img[l, :, :, :][src_mask[l, :, :] == 0, :] = 0
        else:
            # normalize => crop
            # normailize
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
            ref_img -= (0.485, 0.456, 0.406)
            ref_img /= (0.229, 0.224, 0.225)

            ## source
            #for l in range(self.L):
            #    # normalize => crop
            #    # normailize
            #    src_img[l, :, :, :] = (
            #        cv2.cvtColor(src_img[l, :, :, :], cv2.COLOR_BGR2RGB) / 255.0
            #    )
            #    src_img[l, :, :, :] -= (0.485, 0.456, 0.406)
            #    src_img[l, :, :, :] /= (0.229, 0.224, 0.225)

        # from H,W,C to C,H,W
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)

        # depthanything encoder
        _, h, w = ref_img.shape
        #_, _, h, w = ref_img.shape
        #print("---------------------------")
        #print("ref_img.shape: ", ref_img.shape)
        input_tensor, _ =  self.da.image2tensor_simplified(ref_img)
        #print("input_tensor.shape: ", input_tensor.shape)
        with torch.no_grad():
            ref_img = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
        ref_img = F.interpolate(ref_img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, 1024, fH, fW)
        ref_img = ref_img.squeeze(0) # (1024, fH, fW)        


        data_dict["ref_img"] = ref_img

        #src_img = np.transpose(src_img, (0, 3, 1, 2)).astype(np.float32)
        #data_dict["src_img"] = src_img

        return data_dict



class GridSeqDataset_Gibson_metric3d(Dataset):

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
        self.K = np.array([[240, 0, 320], [0, 240, 240], [0, 0, 1]])
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

        # get reference pose
        ref_pose = self.gt_pose[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict["ref_noise"] = 0
        data_dict["ref_pose"] = ref_pose

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

####
        # get depth image
        depth_img = get_depth_image_metric3d(ref_img, K = self.K)
        depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array

        # gravity align depth image
        if self.add_rp:
            r = (np.random.random() - 0.5) * 2 * self.roll
            p = (np.random.random() - 0.5) * 2 * self.pitch
            depth_img, depth_mask = gravity_align_depth(depth_img, r, p, K=self.K)
        else:
            r = 0
            p = 0
            mask = np.ones(list(ref_img.shape[:2]))
            depth_mask = mask.astype(np.uint8)

        data_dict["ref_img"] = depth_img
        data_dict["ref_mask"] = depth_mask

        return data_dict  

####

#        if self.add_rp:
#            # reference
#            # get virtual roll and pitch
#            r = (np.random.random() - 0.5) * 2 * self.roll
#            p = (np.random.random() - 0.5) * 2 * self.pitch
#            # generate mask
#            mask = np.ones(list(ref_img.shape[:2]))
#            #mask = gravity_align(mask, r, p, visualize=False, mode=1)
#            mask = gravity_align(mask, r, p, mode=1)
#            mask[mask < 1] = 0
#            ref_mask = mask.astype(np.uint8)
#            data_dict["ref_mask"] = ref_mask
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
#        else:
#            # normalize => crop
#            # normailize
#            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
#            ref_img -= (0.485, 0.456, 0.406)
#            ref_img /= (0.229, 0.224, 0.225)
#
#        # from H,W,C to C,H,W
#        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
#        data_dict["ref_img"] = ref_img
#
#        return data_dict



class GridSeqDataset_Gibson_metric3d_offline(Dataset):

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
        self.K = np.array([[240, 0, 320], [0, 240, 240], [0, 0, 1]])
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

        # get reference pose
        ref_pose = self.gt_pose[scene_idx][idx_within_scene * (self.L + 1) + self.L]
        data_dict["ref_noise"] = 0
        data_dict["ref_pose"] = ref_pose

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

####

        # load depth image
        depth_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "depth",
            #str(idx_within_scene).zfill(5) + "-" + str(self.L) + ".png",
            str(idx_within_scene).zfill(5) + "-0" + ".npy"
        )
        depth_img = np.load(depth_path).astype(np.float32)

        ## get depth image
        #depth_img = get_depth_image_metric3d(ref_img, K = self.K)
        #depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array

        # gravity align depth image
        if self.add_rp:
            r = (np.random.random() - 0.5) * 2 * self.roll
            p = (np.random.random() - 0.5) * 2 * self.pitch
            depth_img, depth_mask = gravity_align_depth(depth_img, r, p, K=self.K)
        else:
            r = 0
            p = 0
            mask = np.ones(list(ref_img.shape[:2]))
            depth_mask = mask.astype(np.uint8)

        data_dict["ref_img"] = depth_img
        data_dict["ref_mask"] = depth_mask

        return data_dict






















        
#        # align image to gravity
#        roll = ref_euler_angles[0]
#        pitch = ref_euler_angles[1]
#        ref_img = gravity_align(ref_img, r=pitch, p=-(roll+np.pi/2),  mode=1, K=self.K)
#
#        # generate mask
#        mask = np.ones(list(ref_img.shape[:2]))
#        #mask = gravity_align(mask, r, p, mode=1)
#        mask = gravity_align(mask, r=pitch, p=-(roll+np.pi/2),  mode=1, K=self.K)
#        mask[mask < 1] = 0
#        ref_mask = mask.astype(np.uint8)
#        data_dict["ref_mask"] = ref_mask
#
#        # normalize => crop
#        # normalize
#        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
#        ref_img -= (0.485, 0.456, 0.406)
#        ref_img /= (0.229, 0.224, 0.225)
#        # crop
#        ref_img[ref_mask == 0, :] = 0
#
#
#        if False: #self.add_rp:
#            # reference
#            # get virtual roll and pitch
#            r = (np.random.random() - 0.5) * 2 * self.roll
#            p = (np.random.random() - 0.5) * 2 * self.pitch
#            # generate mask
#            mask = np.ones(list(ref_img.shape[:2]))
#            #mask = gravity_align(mask, r, p, visualize=False, mode=1)
#            mask = gravity_align(mask, r, p, mode=1)
#            mask[mask < 1] = 0
#            ref_mask = mask.astype(np.uint8)
#            data_dict["ref_mask"] = ref_mask
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
#        elif False: #else:
#            # normalize => crop
#            # normailize
#            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
#            ref_img -= (0.485, 0.456, 0.406)
#            ref_img /= (0.229, 0.224, 0.225)
#
#
#        # from H,W,C to C,H,W
#        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
#        #print(f"Final ref_img shape: {ref_img.shape}")  # Debug statement
#        data_dict["ref_img"] = ref_img
#        
#        data_dict["scene_name"] = scene_name
#
#        return data_dict





# class GridSeqDataset_hge_customized_cropped_metric3d(Dataset):
# 
#     def __init__(
#         self,
#         dataset_dir,
#         scene_names,
#         L,
#         depth_dir=None,
#         depth_suffix="depth40",
#         add_rp=False,
#         roll=0,
#         pitch=0,
#         start_scene=None,
#         end_scene=None,
#     ):
#         super().__init__()
#         self.dataset_dir = dataset_dir
#         self.scene_names = scene_names
#         self.start_scene = start_scene
#         self.end_scene = end_scene
#         self.L = L
#         self.depth_dir = depth_dir
#         self.depth_suffix = depth_suffix
#         self.add_rp = add_rp
#         self.roll = roll
#         self.pitch = pitch
#         self.scene_start_idx = []
#         self.gt_depth = []
#         self.gt_pose = []
#         self.rgbs = []
#         self.K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])
#         self.gt_euler_angles = []
#         self.metric3d_input_size = (616, 1064) # for vit model
#         self.metric3d_model_name = "ViT-Small" #"ViT-Large"
#         self.min_depth = 0.0
#         self.max_depth = 100.0
#         self.load_scene_start_idx_and_depths_and_poses()
#         if start_scene is None:
#             self.N = self.scene_start_idx[-1]
#         else:
#             # compute N
#             self.N = (
#                 self.scene_start_idx[self.end_scene + 1]
#                 - self.scene_start_idx[self.start_scene]
#             )
# 
#     def __len__(self):
#         return self.N
# 
#     def load_scene_start_idx_and_depths_and_poses(self):
#         self.scene_start_idx.append(0)
#         start_idx = 0
#         print("self.scene_names: ", self.scene_names)
#         for scene in self.scene_names:
#             print("scene: ", scene)
#             # get depths
#             if self.depth_dir == None:
#                 depth_file = os.path.join(
#                     self.dataset_dir, scene, self.depth_suffix + ".txt"
#                 )
#             else:
#                 depth_file = os.path.join(
#                     self.depth_dir, scene, self.depth_suffix + ".txt"
#                 )
# 
#             # read depth
#             with open(depth_file, "r") as f:
#                 depths_txt = [line.strip() for line in f.readlines()]
# 
#             # read pose
#             pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
#             with open(pose_file, "r") as f:
#                 poses_txt = [line.strip() for line in f.readlines()]
# 
#             # read euler angles
#             euler_angles_file = os.path.join(self.dataset_dir, scene, "euler_angles.txt")
#             with open(euler_angles_file, "r") as f:
#                 euler_angles_txt = [line.strip() for line in f.readlines()]
# 
#             traj_len = len(poses_txt)
#             scene_depths = []
#             scene_poses = []
#             scene_euler_angles = []
#             for state_id in range(traj_len):
#                 # get depth
#                 depth = depths_txt[state_id].split(" ")
#                 depth = np.array([float(d) for d in depth]).astype(np.float32)
#                 scene_depths.append(depth)
# 
#                 # get pose
#                 pose = poses_txt[state_id].split(" ")
#                 pose = np.array([float(s) for s in pose]).astype(np.float32)
#                 scene_poses.append(pose)
# 
#                 # get roll and pitch
#                 euler_angles = euler_angles_txt[state_id].split(" ")
#                 euler_angles = np.array([float(s) for s in euler_angles]).astype(np.float32)
#                 scene_euler_angles.append(euler_angles)
#                 #print("euler_angles: ", euler_angles)
# 
#             #print("scene_euler_angles: ", scene_euler_angles)
# 
#             start_idx += traj_len // (self.L + 1)
#             self.scene_start_idx.append(start_idx)
#             self.gt_depth.append(scene_depths)
#             self.gt_pose.append(scene_poses)
#             self.gt_euler_angles.append(scene_euler_angles)
# 
# 
#     def __getitem__(self, idx):
#         """
#         data_dict:
#             "ref_img": (3, H, W)
#             "ref_pose": (3)
#             "src_img": (L, 3, H, W)
#             "src_pose": (L, 3)
#             "ref_mask": (H, W)
#             "src_mask": (L, H, W)
#             "ref_depth": (fW), ground truth depth of the reference img
#             "src_depth": (L, fW), ground truth depth of the source img
#         """
#         if self.start_scene is not None:
#             idx += self.scene_start_idx[self.start_scene]
# 
#         # get the scene name according to the idx
#         scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
#         scene_name = self.scene_names[scene_idx]
# 
#         # get idx within scene
#         idx_within_scene = idx - self.scene_start_idx[scene_idx]
# 
#         # get reference depth
#         ref_depth = self.gt_depth[scene_idx][idx_within_scene * (self.L + 1) + self.L]
#         data_dict = {"ref_depth": ref_depth}
# 
#         # get reference pose
#         ref_pose = self.gt_pose[scene_idx][idx_within_scene * (self.L + 1) + self.L]
#         data_dict["ref_noise"] = 0
#         data_dict["ref_pose"] = ref_pose
# 
#         # get reference euler angles
#         ref_euler_angles = self.gt_euler_angles[scene_idx][idx_within_scene * (self.L + 1) + self.L]
#         data_dict["euler_angles"] = ref_euler_angles
# 
#         # get reference image
#         image_path = os.path.join(
#             self.dataset_dir,
#             scene_name,
#             "rgb",
#             #str(idx_within_scene).zfill(5) + "-" + str(self.L) + ".png",
#             str(idx_within_scene).zfill(5) + "-0" + ".jpg",
#         )
#         #ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(
#         #    np.float32
#         #)  # (H, W, 3)
#         rgb_file = image_path
#         rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]
#         #print(f"Original RGB shape: {rgb_origin.shape}")  # Debug statement
# 
#         model_name = self.metric3d_model_name
#         intrinsic = [self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]] #self.K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])
#         
#         #### adjust input size to fit pretrained model
#         # keep ratio resize
#         input_size = self.metric3d_input_size
#         # input_size = (544, 1216) # for convnext model
#         h, w = rgb_origin.shape[:2]
#         scale = min(input_size[0] / h, input_size[1] / w)
#         rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
#         #print(f"Resized RGB shape: {rgb.shape}")  # Debug statement
#         # remember to scale intrinsic, hold depth
#         intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
#         # padding to input_size
#         padding = [123.675, 116.28, 103.53]
#         h, w = rgb.shape[:2]
#         pad_h = input_size[0] - h
#         pad_w = input_size[1] - w
#         pad_h_half = pad_h // 2
#         pad_w_half = pad_w // 2
#         rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
#         pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
#         #print(f"Padded RGB shape: {rgb.shape}")  # Debug statement
#         #### normalize
#         mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
#         std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
#         rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
#         rgb = torch.div((rgb - mean), std)
#         rgb = rgb[None, :, :, :].cuda()
#         #print(f"Normalized RGB shape: {rgb.shape}")  # Debug statement
#         ###################### canonical camera space ######################        
#         # inference
#         if model_name == "ViT-Small":
#             model = metric3d_vit_small(pretrain=True)
#         elif model_name == "ViT-Large":
#             model = metric3d_vit_large(pretrain=True)
#         elif model_name == "ViT-giant2":
#             model = metric3d_vit_giant2(pretrain=True)
#         model.cuda().eval()
#         #start_time = time.time()
#         with torch.no_grad():
#             pred_depth, confidence, output_dict = model.inference({'input': rgb})
#         #end_time = time.time()
#         #execution_time = end_time - start_time
#         #print(f"Execution time: {execution_time} seconds")
# 
#         # un pad
#         pred_depth = pred_depth.squeeze()
#         pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
#         
#         # upsample to original size
#         pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
#         ###################### canonical camera space ######################
# 
#         #### de-canonical transform
#         canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
#         pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
#         pred_depth = torch.clamp(pred_depth, 0, 300)
#         ref_img = pred_depth.cpu().numpy()
# 
#         #### align image to gravity
#         roll = ref_euler_angles[0]
#         pitch = ref_euler_angles[1]
#         ref_img = gravity_align(ref_img, r=pitch, p=-(roll+np.pi/2),  mode=1, K=self.K)
# 
#         # generate mask
#         mask = np.ones(list(ref_img.shape[:2]))
#         #mask = gravity_align(mask, r, p, mode=1)
#         mask = gravity_align(mask, r=pitch, p=-(roll+np.pi/2),  mode=1, K=self.K)
#         mask[mask < 1] = 0
#         ref_mask = mask.astype(np.uint8)
#         data_dict["ref_mask"] = ref_mask
# 
#         # normalize => crop
#         # normalize
#         #ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
#         #ref_img -= (0.485, 0.456, 0.406)
#         #ref_img /= (0.229, 0.224, 0.225)
#         ref_img = (ref_img - self.min_depth) / (self.max_depth - self.min_depth)
# 
#         # crop
#         ref_img[ref_mask == 0,] = 0
#         ref_img = np.expand_dims(ref_img, axis=0)
#         #ref_img[ref_mask == 0, :] = 0
#         #print(f"Final ref_img shape: {ref_img.shape}")  # Debug statement
# 
# 
#         if False: #self.add_rp:
#             # reference
#             # get virtual roll and pitch
#             r = (np.random.random() - 0.5) * 2 * self.roll
#             p = (np.random.random() - 0.5) * 2 * self.pitch
#             # generate mask
#             mask = np.ones(list(ref_img.shape[:2]))
#             #mask = gravity_align(mask, r, p, visualize=False, mode=1)
#             mask = gravity_align(mask, r, p, mode=1)
#             mask[mask < 1] = 0
#             ref_mask = mask.astype(np.uint8)
#             data_dict["ref_mask"] = ref_mask
# 
#             # normalize => crop
#             # normailize
#             ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
#             ref_img -= (0.485, 0.456, 0.406)
#             ref_img /= (0.229, 0.224, 0.225)
# 
#             # crop
#             ref_img[ref_mask == 0, :] = 0
# 
#         elif False: #else:
#             # normalize => crop
#             # normailize
#             ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0
#             ref_img -= (0.485, 0.456, 0.406)
#             ref_img /= (0.229, 0.224, 0.225)
# 
# 
#         # from H,W,C to C,H,W
#         #ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
#         data_dict["ref_img"] = ref_img
#         
#         data_dict["scene_name"] = scene_name
# 
#         return data_dict




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





class S3DDataset(Dataset):

    def __init__(self, dataset_dir, scene_range, depth_suffix="depth40", return_original=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_range = scene_range
        self.depth_suffix = depth_suffix
        self.scene_names = []
        self.scene_start_idx = []
        self.gt_depths = []
        self.gt_poses = []
        self.return_original = return_original
        self.load_scene_start_idx_and_rays()
        self.N = self.scene_start_idx[-1]


    def __len__(self):
        return self.N

    
    def load_scene_start_idx_and_rays(self):
        self.scene_start_idx.append(0)
        start_idx = 0 
        for scene in sorted(os.listdir(self.dataset_dir)):
            if not os.path.isdir(os.path.join(self.dataset_dir, scene)):
                continue
            
            if int(scene[-5:]) > self.scene_range[1] or int(scene[-5:]) < self.scene_range[0]:
                continue
            
            self.scene_names.append(scene)
            # get depth
            depth_file = os.path.join(self.dataset_dir, scene, self.depth_suffix+".txt")
                
            # read depth
            with open(depth_file, "r") as f:
                depth_txt = [line.strip() for line in f.readlines()]
            
            traj_len = len(depth_txt)
            scene_depths = []
            for state_id in range(traj_len):

                # get depth
                depth = depth_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth])
                scene_depths.append(depth)

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses_map.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            scene_poses = []
            for state_id in range(traj_len):
                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)


            start_idx += traj_len
            self.scene_start_idx.append(start_idx)
            self.gt_depths.append(scene_depths)
            self.gt_poses.append(scene_poses)
        
        self.gt_depths = np.concatenate([np.stack(scene_depths) for scene_depths in self.gt_depths])
        self.gt_poses = np.concatenate([np.stack(scene_poses) for scene_poses in self.gt_poses])
    
    def __getitem__(self, idx):
        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # read the image
        image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            #"rgb",
            "imgs",
            str(idx_within_scene).zfill(3) + ".png",
        )
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # align the image here
        # hardcoded intrinsics from fov
        K = np.array([[320/np.tan(0.698132), 0, 320],
                    [0, 180/np.tan(0.440992), 180],
                    [0, 0, 1]], dtype=np.float32)
        
        # align the image
        if self.return_original:
            original_img = img.copy()
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) / 255.0
        # normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img -= (0.485, 0.456, 0.406)
        img /= (0.229, 0.224, 0.225)

        
        # gravity align
        img = gravity_align(img, r=self.gt_poses[idx][-2], p=self.gt_poses[idx][-1], K=K)#, visualize=False)

        # compute the attention mask
        mask = np.ones(list(img.shape[:2]))
        mask = gravity_align(mask, r=self.gt_poses[idx][-2], p=self.gt_poses[idx][-1], K=K)#, visualize=False)
        mask[mask < 1] = 0
        mask = mask.astype(np.uint8)
        img = np.transpose(img, [2, 0, 1]).astype(np.float32) # (C, H, W)
       

        #data_dict = {"gt_rays": self.gt_depths[idx], "ref_img": img, "ref_mask":mask}
        data_dict = {"ref_pose": self.gt_poses[idx], "ref_depth": self.gt_depths[idx], "ref_img": img, "ref_mask":mask}
        if self.return_original:
            data_dict["original_img"] = original_img
        return data_dict




class S3DDataset_depthanything(Dataset):

    def __init__(self, dataset_dir, scene_range, depth_suffix="depth40", return_original=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_range = scene_range
        self.depth_suffix = depth_suffix
        self.scene_names = []
        self.scene_start_idx = []
        self.gt_depths = []
        self.gt_poses = []
        self.return_original = return_original
        self.load_scene_start_idx_and_rays()
        self.N = self.scene_start_idx[-1]

        ####
        # DepthAnything Encoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        encoder =  'vits' # or 'vitl', 'vitb', 'vits'
        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        self.da = DepthAnythingV2(**model_configs[encoder])
        self.da.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
        self.da = self.da.to(device)  # Move the model to GPU
        self.da.eval()
        ####


    def __len__(self):
        return self.N

    
    def load_scene_start_idx_and_rays(self):
        self.scene_start_idx.append(0)
        start_idx = 0 
        for scene in sorted(os.listdir(self.dataset_dir)):
            if not os.path.isdir(os.path.join(self.dataset_dir, scene)):
                continue
            
            if int(scene[-5:]) > self.scene_range[1] or int(scene[-5:]) < self.scene_range[0]:
                continue
            
            self.scene_names.append(scene)
            # get depth
            depth_file = os.path.join(self.dataset_dir, scene, self.depth_suffix+".txt")
                
            # read depth
            with open(depth_file, "r") as f:
                depth_txt = [line.strip() for line in f.readlines()]
            
            traj_len = len(depth_txt)
            scene_depths = []
            for state_id in range(traj_len):

                # get depth
                depth = depth_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth])
                scene_depths.append(depth)

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses_map.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            scene_poses = []
            for state_id in range(traj_len):
                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)


            start_idx += traj_len
            self.scene_start_idx.append(start_idx)
            self.gt_depths.append(scene_depths)
            self.gt_poses.append(scene_poses)
        
        self.gt_depths = np.concatenate([np.stack(scene_depths) for scene_depths in self.gt_depths])
        self.gt_poses = np.concatenate([np.stack(scene_poses) for scene_poses in self.gt_poses])
    
    def __getitem__(self, idx):
        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # read the image
        image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            #"rgb",
            "imgs",
            str(idx_within_scene).zfill(3) + ".png",
        )
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # align the image here
        # hardcoded intrinsics from fov
        K = np.array([[320/np.tan(0.698132), 0, 320],
                    [0, 180/np.tan(0.440992), 180],
                    [0, 0, 1]], dtype=np.float32)
        
        # align the image
        if self.return_original:
            original_img = img.copy()
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) / 255.0
        # normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img -= (0.485, 0.456, 0.406)
        img /= (0.229, 0.224, 0.225)

        
        # gravity align
        img = gravity_align(img, r=self.gt_poses[idx][-2], p=self.gt_poses[idx][-1], K=K)#, visualize=False)

        # compute the attention mask
        mask = np.ones(list(img.shape[:2]))
        mask = gravity_align(mask, r=self.gt_poses[idx][-2], p=self.gt_poses[idx][-1], K=K)#, visualize=False)
        mask[mask < 1] = 0
        mask = mask.astype(np.uint8)
        img = np.transpose(img, [2, 0, 1]).astype(np.float32) # (C, H, W)
       

### 22.10.2024
        # depthanything encoder
        _, h, w = img.shape
        #_, _, h, w = ref_img.shape
        #print("---------------------------")
        #print("ref_img.shape: ", ref_img.shape)
        input_tensor, _ =  self.da.image2tensor_simplified(img)
        #print("input_tensor.shape: ", input_tensor.shape)
        with torch.no_grad():
            img = self.da.pretrained.get_intermediate_layers(input_tensor, reshape=True)[0]
        img = F.interpolate(img, size=(h//16, w//16), mode='bilinear', align_corners=True) # (1, 1024, fH, fW)
        img = img.squeeze(0) # (1024, fH, fW) 
###


        #data_dict = {"gt_rays": self.gt_depths[idx], "ref_img": img, "ref_mask":mask}
        data_dict = {"ref_pose": self.gt_poses[idx], "ref_depth": self.gt_depths[idx], "ref_img": img, "ref_mask":mask}
        if self.return_original:
            data_dict["original_img"] = original_img
        return data_dict