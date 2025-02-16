#!/usr/bin/env python
# coding: utf-8


import os
from pathlib import Path

root_directory = "/cluster/home/wueestm/f3loc/"
os.chdir(root_directory)
root_directory
Path.cwd()


import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from utils.utils import gravity_align, gravity_align_depth, gravity_align_normal, gravity_align_segmentation
from metric3d.hubconf import *
from src.depth_image_functions import *
import yaml
from attrdict import AttrDict



dataset = "gibson_t" # "hge_customized_complete" #"gibson_f" #"hge_customized_cropped"
encoder = "metric3d_large" # "metric3d_small" # 
if dataset == "hge_customized_complete":
    dataset_path = "/cluster/project/cvg/data/lamar/HGE_customized_complete" #"/cluster/project/cvg/data/lamar/HGE_customized_cropped"
elif (dataset == "gibson_f") or (dataset == "gibson_t"):
    dataset_path= "/cluster/project/cvg/data/gibson/Gibson_Floorplan_Localization_Dataset"
dataset_dir = os.path.join(dataset_path, dataset)
split_file = os.path.join(dataset_dir, "split.yaml")
with open(split_file, "r") as f:
    split = AttrDict(yaml.safe_load(f))


if dataset == "hge_customized_complete":
    scene_names = split.train + split.test + split.rest 
elif (dataset == "gibson_f") or (dataset == "gibson_t"):
    scene_names = split.test # split.train + split.val + split.test #split.rest 

print("-------------------------")
print("type: ", type(scene_names))
print(scene_names)
print("-------------------------")

if dataset == "hge_customized_complete":
    start_scene = None
    end_scene = None
    L = 0
    scene_start_idx = []
    gt_depth = []
    gt_pose = []
    rgbs = []
    K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])
elif (dataset == "gibson_f"):
    start_scene = None
    end_scene = None
    L = 3
    scene_start_idx = []
    gt_depth = []
    gt_pose = []
    K = np.array([[240, 0, 320], [0, 240, 240], [0, 0, 1]])
elif (dataset =="gibson_t"):
    start_scene = None
    end_scene = None
    L = 0
    scene_start_idx = []
    gt_depth = []
    gt_pose = []
    K = np.array([[240, 0, 320], [0, 240, 240], [0, 0, 1]])



scene_start_idx.append(0)
start_idx = 0
print("scene_names: ", scene_names)
for scene in scene_names:
    print("scene: ", scene)

    # read pose
    pose_file = os.path.join(dataset_dir, scene, "poses.txt")
    with open(pose_file, "r") as f:
        poses_txt = [line.strip() for line in f.readlines()]
    traj_len = len(poses_txt)

    start_idx += traj_len // (L + 1)
    scene_start_idx.append(start_idx)



if start_scene is None:
    N = scene_start_idx[-1]
else:
    # compute N
    N = (
        scene_start_idx[end_scene + 1]
        - scene_start_idx[start_scene]
    )
print(scene_start_idx)


for idx in range(N):
#for idx in range(10):
    if start_scene is not None:
        idx += scene_start_idx[start_scene]

    print("idx: ", idx, "/", N)


    # get the scene name according to the idx
    scene_idx = np.sum(idx >= np.array(scene_start_idx)) - 1
    scene_name = scene_names[scene_idx]
    print("scene_name: ", scene_name)

    # get idx within scene
    idx_within_scene = idx - scene_start_idx[scene_idx]

    # get reference image
    if dataset == "hge_customized_complete":
        image_path = os.path.join(
            dataset_dir,
            scene_name,
            "rgb",
            str(idx_within_scene).zfill(5) + "-0" + ".jpg",
        )
    elif (dataset == "gibson_f"):
        image_path = os.path.join(
            dataset_dir,
            scene_name,
            "rgb",
            str(idx_within_scene).zfill(5) + "-" + str(L) + ".png",
        )
    elif (dataset == "gibson_t"):
        image_path = os.path.join(
            dataset_dir,
            scene_name,
            "rgb",
            str(idx_within_scene).zfill(5) + ".png",
        )        

    ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(
        np.float32
    )  # (H, W, 3)

    # get and save depth image
    if encoder == "metric3d_large":
        depth_img = get_depth_image_metric3d(ref_img, K = K)
        depth_folder = os.path.join(dataset_dir, scene_name, "depth")
    elif encoder == "metric3d_small":
        depth_img = get_depth_image_metric3d_small(ref_img, K = K)
        depth_folder = os.path.join(dataset_dir, scene_name, "depth_small")
    depth_img = depth_img.cpu().numpy()  # Move to CPU and convert to numpy array
    depth_img = depth_img.astype(np.float16)    
    depth_path = os.path.join(depth_folder, str(idx_within_scene).zfill(5) + "-0" + ".npy")
    os.makedirs(depth_folder, exist_ok=True)
    np.save(depth_path, depth_img)

    print("Depth image done.")