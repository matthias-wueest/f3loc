#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import tqdm
import yaml
from attrdict import AttrDict
from torch.utils.data import DataLoader
import matplotlib.image as mpimg 

from modules.comp.comp_d_net_pl import *
from modules.mono.depth_net_pl import *
from modules.mv.mv_depth_net_pl import *
from utils.data_utils import *
from utils.localization_utils import *
#from utils.data_utils import TrajDataset_hge_customized_cropped

from src.helper_functions import *
from src.depth_aggregation import *


# In[ ]:







# In[2]:


net_type = "d"
dataset = "gibson_t" # "gibson_g" # 
#dataset = "hge_customized_cropped"
#dataset = "hge_customized_complete"
model = "depthanything" #"metric3d_attention" # "gt" # "f3loc_own" # "metric3d_percentile_95" # 
uncertainty_bool = False
dt = 1 # 2 # 5 # 
traj_len = 100#50 #35 #15# 15# 100#8#100#50

if dataset == "gibson_t":
    dataset_path = "/cluster/project/cvg/data/gibson/Gibson_Floorplan_Localization_Dataset"
    #evol_path = "./evol_path/gibson_f/gt" #evol_path = "./evol_path/gibson_f/mono"
    evol_path = "./evol_path/gibson_t/depthanything_vitl" +  "traj_len_" + str(traj_len) + "_dt_" + str(dt) #evol_path = "./evol_path/gibson_f/mono"
    evol_path = "./evol_path/gibson_t/paper_observation_likelihood_depthanything_vitl" +  "traj_len_" + str(traj_len) + "_dt_" + str(dt) #evol_path = "./evol_path/gibson_f/mono"
    desdf_resolution = 0.1
    orn_slice = 36
elif dataset == "hge_customized_cropped":
    dataset_path = "/cluster/project/cvg/data/lamar/HGE_customized_cropped"
    evol_path = "./evol_path/hge_customized_cropped/gt"
    desdf_resolution = 0.1
    orn_slice = 36
    #desdf_resolution = 1
    #orn_slice = 360
elif dataset =="hge_customized_complete":
    dataset_path = "/cluster/project/cvg/data/lamar/HGE_customized_complete"
    evol_path = "./evol_path/hge_customized_complete/gt/gravity_align"
    desdf_resolution = 0.1
    orn_slice = 36


ckpt_path = "./logs"
#checkpoint_path = "./tb_logs/my_model/version_42/checkpoints/epoch=19-step=19100.ckpt"
#checkpoint_path = "./tb_logs/my_model/version_48/checkpoints/epoch=26-step=25785.ckpt"
#checkpoint_path  = "./logs/mono.ckpt"
#checkpoint_path = "./tb_logs/my_model/version_114/checkpoints/epoch=19-step=52120.ckpt"
#checkpoint_path = "./tb_logs/my_model/version_116/checkpoints/epoch=49-step=130300.ckpt"
#checkpoint_path = "./tb_logs/my_model/version_117/checkpoints/epoch=1-step=5212.ckpt"
#checkpoint_path = "./tb_logs/my_model/version_117/checkpoints/epoch=2-step=7818.ckpt"
#checkpoint_path = "./tb_logs/my_model/version_162/checkpoints/epoch=2-step=15633.ckpt"
#checkpoint_path = "./tb_logs/my_model/version_224/checkpoints/last.ckpt" #vitb
#checkpoint_path = "./tb_logs/my_model/version_230/checkpoints/epoch=46-step=81639.ckpt"
checkpoint_path = "./tb_logs/my_model/version_162/checkpoints/epoch=38-step=203229copy.ckpt"




# In[3]:


# get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("======= USING DEVICE : ", device, " =======")


# In[4]:


# paths
dataset_dir = os.path.join(dataset_path, dataset)
depth_dir = dataset_dir#args.dataset_path
log_dir = ckpt_path
desdf_path = os.path.join(dataset_path, "desdf")
evol_path = evol_path


# In[5]:


# depth file suffix
if (net_type == "d") & ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
    depth_suffix = "depth90"
elif net_type == "d":
    depth_suffix = "depth40"
else:
    depth_suffix = "depth160"


# In[6]:


# instantiate dataset
traj_l = traj_len
split_file = os.path.join(dataset_dir, "split.yaml")
with open(split_file, "r") as f:
    split = AttrDict(yaml.safe_load(f))

if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):

    test_set = TrajDataset_hge_customized_cropped_gravity_align(
        dataset_dir,
        split.test,
        L=traj_l,
        depth_dir=depth_dir,
        depth_suffix=depth_suffix,
        add_rp=False,
        roll=0,
        pitch=0,
        without_depth=False, #without_depth=True,  
    )
#    test_set = TrajDataset_hge_customized_metric3d(
#        dataset_dir,
#        split.test,
#        L=traj_l,
#        depth_dir=depth_dir,
#        depth_suffix=depth_suffix,
#        add_rp=False,
#        roll=0,
#        pitch=0,
#        without_depth=False, #without_depth=True,  
#    )

else:

    if model == "f3loc_own":
        test_set = TrajDataset(
            dataset_dir,
            split.test,
            L=traj_l,
            depth_dir=depth_dir,
            depth_suffix=depth_suffix,
            add_rp=False,
            roll=0,
            pitch=0,
            without_depth=False, #without_depth=True,  
        )

    elif model == "metric3d_attention":
        test_set = TrajDataset_gibson_metric3d(
            dataset_dir,
            split.test,
            L=traj_l,
            depth_dir=depth_dir,
            depth_suffix=depth_suffix,
            add_rp=False,
            roll=0,
            pitch=0,
            without_depth=False, #without_depth=True,  
        )    

    elif model == "depthanything":
        test_set = TrajDataset_gibson_depthanything(
            dataset_dir,
            split.test,
            L=traj_l,
            depth_dir=depth_dir,
            depth_suffix=depth_suffix,
            add_rp=False,
            roll=0,
            pitch=0,
            without_depth=False, #without_depth=True,  
        )    


# In[ ]:







# In[7]:


# logs
log_error = False
log_timing = False
log_extended = False


# In[8]:


# # parameters
L = 0 #3  # number of the source frames
D = 128  # number of the depth planes
d_min = 0.1  # minimum depth
d_max = 15.0  # maximum depth
d_hyp = -0.2  # depth transform (uniform sampling in d**d_hyp)
F_W = 3 / 8  # camera intrinsic, focal length / image width
trans_thresh = 0.005  # translation threshold (variance) if using comp_s

# # parameters
# # L = 0  # number of the source frames
# D = 128  # number of depth planes
# d_min = 0.1  # minimum depth
# d_max = 100  # maximum depth
# d_hyp = 0.2  # depth transform (uniform sampling in d**d_hyp)






# In[9]:


# models
if net_type == "mvd" or net_type == "comp_s":
    # instaciate model
    mv_net = mv_depth_net_pl.load_from_checkpoint(
        checkpoint_path=os.path.join(log_dir, "mv.ckpt"),
        D=D,
        d_min=d_min,
        d_max=d_max,
        d_hyp=d_hyp,
    ).to(device)
if net_type == "d" or net_type == "comp_s":
    # instaciate model
    #checkpoint_path = os.path.join(log_dir, "mono.ckpt")

    if uncertainty_bool == False:
        if model == "f3loc_own":
            d_net = depth_net_pl.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                d_min=d_min,
                d_max=d_max,
                d_hyp=d_hyp,
                D=D,
            ).to(device)
        elif model == "metric3d_attention":
            d_net = depth_net_metric3d_pl.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                d_min=d_min,
                d_max=d_max,
                d_hyp=d_hyp,
                D=D,
                ).to(device)
        elif (model == "depthanything"):
            d_net = depth_net_depthanything_pl.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                d_min=d_min,
                d_max=d_max,
                d_hyp=d_hyp,
                D=D,
            ).to(device)

    else:
        d_net = depth_net_uncertainty_pl.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
            D=D,
        ).to(device)

if net_type == "comp":
    mv_net_pl = mv_depth_net_pl(D=D, d_hyp=d_hyp, F_W=F_W)
    mono_net_pl = depth_net_pl(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D, F_W=F_W)
    comp_net = comp_d_net_pl.load_from_checkpoint(
        checkpoint_path=os.path.join(log_dir, "comp.ckpt"),
        mv_net=mv_net_pl.net,
        mono_net=mono_net_pl.encoder,
        L=L,
        d_min=d_min,
        d_max=d_max,
        d_hyp=d_hyp,
        D=D,
        F_W=F_W,
        use_pred=True,
    ).to(device)
#     comp_net.eval()  # this is needed to disable batchnorm


# In[10]:


# get desdf for the scene
print("load desdf ...")
desdfs = {}

if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
    ### Improved i.t.o. memory
    scene = test_set.scene_names[0]
    desdfs["desdf"] = np.load(
            os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True
        ).item()
    ###
else:
    for scene in tqdm.tqdm(test_set.scene_names):
        desdfs[scene] = np.load(
            os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True
        ).item()
    #    desdfs[scene]["desdf"][desdfs[scene]["desdf"] > 10] = 10  # truncate
    print("desdf shape: ", desdfs[scene]["desdf"].shape)



if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):

    # Correspondences for calibration
    floorplan_correspondences = np.array([[1516, 490.5], [1532, 2162], [350, 1515], [391, 135]])
    trajectory_correspondences = np.array([[-12.600, 10.103], [77.981, 9.882], [42.641, -54.348], [-33.099, -51.229]])

    # Test the transformation on the source points
    pts_src = trajectory_correspondences
    pts_dst = floorplan_correspondences
    affine_matrix = find_affine_transform(pts_src, pts_dst)
    transformed_pts = np.array([apply_affine_transformation(pt, affine_matrix) for pt in pts_src])
    pixel_per_meter = (affine_matrix[0,1] + affine_matrix[1,0]) / 2

    print("Affine transformation matrix:\n", affine_matrix)
    print("Destination points:\n", pts_dst)
    print("Transformed points:\n", transformed_pts)
    print("pixel_per_meter:", pixel_per_meter)




# In[14]:


# get the ground truth pose file
print("load poses and maps ...")
maps = {}
gt_poses = {}
for scene in tqdm.tqdm(test_set.scene_names):
    # load map
    occ = cv2.imread(os.path.join(dataset_dir, scene, "map.png"))[:, :, 0]
    maps[scene] = occ
    h = occ.shape[0]
    w = occ.shape[1]

    # single trajectory
    poses = np.zeros([0, 3], dtype=np.float32)
    # get poses
    poses_file = os.path.join(dataset_dir, scene, "poses.txt")

    # read poses
    with open(poses_file, "r") as f:
        poses_txt = [line.strip() for line in f.readlines()]

    traj_len = len(poses_txt)
    traj_len -= traj_len % traj_l
    for state_id in range(traj_len):

        if dataset == "hge_customized_cropped":
            # get pose
            pose_ls = poses_txt[state_id].split(" ")
            pose = np.array(pose_ls, dtype=np.float32)
            # world to map 
            (x,y), th = world_to_map(position_world=pose[:2], 
                                        orientation_world_rad=pose[2], 
                                        affine_matrix=affine_matrix, 
                                        floorplan_correspondences=floorplan_correspondences)
        elif dataset =="hge_customized_complete":
            # get pose
            pose_ls = poses_txt[state_id].split(" ")
            pose = np.array(pose_ls, dtype=np.float32)
            # world to map 
            (x,y), th = world_to_map_hge_complete(position_world=pose[:2], 
                                        orientation_world_rad=pose[2], 
                                        affine_matrix=affine_matrix)
        else:
            # get pose
            pose = poses_txt[state_id].split(" ")
            x = float(pose[0])
            y = float(pose[1])
            th = float(pose[2])
            # from world coordinate to map coordinate
            x = x / 0.01 + w / 2
            y = y / 0.01 + h / 2

        poses = np.concatenate(
            (poses, np.expand_dims(np.array((x, y, th), dtype=np.float32), 0)),
            axis=0,
        )

    gt_poses[scene] = poses


# In[ ]:







# In[ ]:


# original_resolution = 0.01
# desdf_resolution = 0.1
# resolution_ratio = desdf_resolution / original_resolution
# resolution_ratio


# In[ ]:


# original_resolution = 1/pixel_per_meter
# desdf_resolution = 1
# resolution_ratio = desdf_resolution / original_resolution
# resolution_ratio


# In[15]:


# record stats
RMSEs = []
success_10 = []  # Success @ 1m
success_5 = []  # Success @ 0.5m
success_3 = []  # Success @ 0.3m
success_2 = []  # Success @ 0.2m

matching_time = 0
iteration_time = 0
feature_extraction_time = 0
n_iter = 0

####
mean_depth_MAE_ls = []
mean_depth_cos_sim_ls = []
recalls_all_ls = []
####

print("Length Test Set: ", len(test_set))

# loop the over scenes
test_set_ls = [1, 23, 26, 30, 35]
print("length of test_set: ", len(test_set))
for data_idx in tqdm.tqdm(test_set_ls):

    print("Before loading data")
    data = test_set[data_idx]
    print("After loading data")
    # get the scene name according to the data_idx
    scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
    scene = test_set.scene_names[scene_idx]

    # get idx within scene
    idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]

    # get desdf
    if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
        desdf = desdfs["desdf"]
    else:
        desdf = desdfs[scene]
    print("After getting desdf")

    # get reference pose in map coordinate and in scene coordinate
    poses_map = gt_poses[scene][
        idx_within_scene * traj_l : idx_within_scene * traj_l + traj_l, :
    ]

    # map to desdf 
    if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
        original_resolution = 1/pixel_per_meter
        resolution_ratio = desdf_resolution / original_resolution
        gt_pose_desdf = poses_map.copy()
        gt_pose_desdf[:, 0] = (gt_pose_desdf[:, 0] - desdf["l"]) / resolution_ratio
        gt_pose_desdf[:, 1] = (gt_pose_desdf[:, 1] - desdf["t"]) / resolution_ratio
        gt_pose_desdf[:, 2] = gt_pose_desdf[:, 2]
        #masks = torch.tensor(data["masks"], device=device).unsqueeze(0)
    else:
        original_resolution = 0.01
        gt_pose_desdf = poses_map.copy()
        gt_pose_desdf[:, 0] = (gt_pose_desdf[:, 0] - desdf["l"]) / 10
        gt_pose_desdf[:, 1] = (gt_pose_desdf[:, 1] - desdf["t"]) / 10
    print("After map to desdf")

    imgs = torch.tensor(data["imgs"], device=device).unsqueeze(0)
    print(len(data["poses"]))
    poses = torch.tensor(data["poses"], device=device).unsqueeze(0)
    print("tensor poses")

    # set prior as uniform distribution
    prior = torch.tensor(
        np.ones_like(desdf["desdf"]) / desdf["desdf"].size, device=device
    ).to(torch.float32)
    print("prior loaded to gpu")
    pred_poses_map = []

    # record stats extended: per trajectory
    if (evol_path is not None) and (log_extended==True):
        metric_depth_l1_loss_ls = []
        metric_depth_shape_loss_ls = []
        metric_ray_l1_loss_ls = []
        metric_observation_position_err_ls = []
        metric_observation_orientation_err_ls = []
        metric_posterior_position_err_ls = []
        metric_posterior_orientation_err_ls = []
        bf_current_pose_ls = []
        bf_transition_ls = []

    print("start looping over sequences")
    # loop over the sequences
    for t in range(traj_l - L):
        print("t = ", t)
        start_iter = time.time()
        feature_extraction_start = time.time()
        # form input
        input_dict = {}
        if net_type == "mvd" or net_type == "comp" or net_type == "comp_s":
            input_dict.update(
                {
                    "ref_img": imgs[:, t + L, :, :, :],
                    "src_img": imgs[:, t : t + L, :, :, :],
                    "ref_pose": poses[:, t + L, :],
                    "src_pose": poses[:, t : t + L, :],
                    "ref_mask": None,  # no masks because the dataset has zero roll pitch
                    "src_mask": None,  # no masks because the dataset has zero roll pitch
                }
            )
        if net_type == "d" or net_type == "comp_s":
            
            if (model == "metric3d_percentile_95") or (model == "metric3d_attention") or (model == "depthanything"):
                input_dict.update(
                            {
                                "img": imgs[:, t + L, :, :],
                                "mask": None,  # no masks because the dataset has zero roll pitch
                            }
                        )
            else:
                input_dict.update(
                    {
                        "img": imgs[:, t + L, :, :, :],
                        "mask": None,  # no masks because the dataset has zero roll pitch
                        #"mask": masks[:, t + L, :, :],
                    }
                )

            # # depth images without masks
            # input_dict.update(
            #     {
            #         "img": imgs[:, t + L, :, :],
            #         "mask": None
            #     }
            # )



        # check which model to use if hardcoded selection
        if net_type == "comp_s":
            # calculate the relative poses
            pose_var = (
                torch.cat(
                    (input_dict["ref_pose"].unsqueeze(1), input_dict["src_pose"]),
                    dim=1,
                )
                .squeeze(0)
                .var(dim=0)[:2]
                .sum()
            )
            if pose_var < trans_thresh:
                use_mv = False
                use_mono = True
            else:
                use_mv = True
                use_mono = False
        #print("Before inference")

        # inference
        if net_type == "mvd" or (net_type == "comp_s" and use_mv):
            pred_dict = mv_net.net(input_dict)
            pred_depths = pred_dict["d"]
            pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()
        elif net_type == "d" or (net_type == "comp_s" and use_mono):

            if uncertainty_bool == False:
                ### Trained model:
                pred_depths, attn_2d, prob = d_net.encoder(
                    input_dict["img"], input_dict["mask"]
                )
                pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()
                ###
            else:
                ### Trained model:
                loc, scale, attn, prob = d_net.encoder(
                    input_dict["img"], input_dict["mask"]
                )
                pred_depths = loc.squeeze(0).detach().cpu().numpy()
                pred_scales = scale.squeeze(0).detach().cpu().numpy()
                ###

            # ### GT:
            # pred_depths = np.array(data["gt_depth"])[t + L,:]
            # ###
            
            # ### Percentile of depth image
            # input_img_np = input_dict["img"].squeeze(0).detach().cpu().numpy()
            # if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
            #     correction_factor = 1.5
            # else:
            #     correction_factor = 1.0
            # pred_depths = correction_factor*get_column_percentile_and_downsample(input_img_np, p=95)

        elif net_type == "comp":
            pred_dict = comp_net.comp_d_net(input_dict)
            pred_depths = pred_dict["d_comp"]
            pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()

        print("Got pred_depths")

        #pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()

        # get rays from depth
        if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
            # fov_desdf = 49#50
            # dv = 360/orn_slice
            # V = fov_desdf / dv
            # pred_rays = get_ray_from_depth(pred_depths, V=V, dv=dv, F_W=1596/1440)
            # pred_rays = torch.tensor(pred_rays, device=device)

            pred_rays = get_ray_from_depth(pred_depths)
            pred_rays = torch.tensor(pred_rays, device=device)

        else:
            if uncertainty_bool == False:
                pred_rays = get_ray_from_depth(pred_depths)
                pred_rays = torch.tensor(pred_rays, device=device)
            else:
                # Laplace
                pred_rays, scales_rays = get_ray_from_depth_uncertainty(pred_depths, pred_scales)
                pred_rays = torch.tensor(pred_rays, device=device)
                scales_rays = torch.tensor(scales_rays, device=device)

        print(pred_rays)
        print(pred_rays.shape)

        feature_extraction_end = time.time()

        matching_start = time.time()
        print("Before localize")

        # use the prediction to localize, produce observation likelihood
        if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
            likelihood, likelihood_2d, likelihood_orn, likelihood_pred = localize_noflip(
                torch.tensor(desdf["desdf"]).to(prior.device),
                pred_rays.to(prior.device),
                return_np=False,
                orn_slice=orn_slice,
                lambd=40
            )
        else:
            if uncertainty_bool == False:
                likelihood, likelihood_2d, likelihood_orn, likelihood_pred = localize(
                    torch.tensor(desdf["desdf"]).to(prior.device),
                    pred_rays.to(prior.device),
                    return_np=False,
                    orn_slice=orn_slice
                )
            else:
                # Laplace
                likelihood, likelihood_2d, likelihood_orn, likelihood_pred = localize_uncertainty(
                    torch.tensor(desdf["desdf"]).to(prior.device),
                    pred_rays.to(prior.device),
                    scales_rays.to(prior.device),
                    return_np=False,
                    orn_slice=orn_slice
                )
            
        matching_end = time.time()

        ## multiply with the prior
        #posterior = prior * likelihood.to(prior.device)
        #posterior = posterior / posterior.sum()

        # multiply with the prior
        if t % dt == 0:
            posterior = prior * likelihood.to(prior.device)
        else:
            posterior = prior
        #posterior = prior * likelihood.to(prior.device)
        epsilon = 1e-10
        posterior = posterior / (posterior.sum() + epsilon)
        
        print(f"prior shape: {prior.shape}")
        print(f"prior range: {prior.min().item()} to {prior.max().item()}")
        print(f"prior sum: {prior.sum()}")
        print(f"likelihood shape: {likelihood.shape}")
        print(f"likelihood range: {likelihood.min().item()} to {likelihood.max().item()}")
        print(f"likelihood sum: {likelihood.sum()}")
        print(f"posterior shape: {posterior.shape}")
        print(f"posterior range: {posterior.min().item()} to {posterior.max().item()}")
        print(f"posterior sum: {posterior.sum()}")


        # reduce the posterior along orientation for 2d visualization
        posterior_2d, orientations = torch.max(posterior, dim=2)

        # compute prior_2d for visualization
        prior_2d, _ = torch.max(prior, dim=2)

        # maximum of the posterior as result
        pose_y, pose_x = torch.where(posterior_2d == posterior_2d.max())
        if pose_y.shape[0] > 1:
            pose_y = pose_y[0].unsqueeze(0)
            pose_x = pose_x[0].unsqueeze(0)
        orn = orientations[pose_y, pose_x]

        # from orientation indices to radians
        orn = orn / orn_slice * 2 * torch.pi
        pose = torch.cat((pose_x, pose_y, orn)).detach().cpu().numpy()

        if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
            pose_in_map = pose.copy()
            pose_in_map[0] = pose_in_map[0] * resolution_ratio  + desdf["l"]
            pose_in_map[1] = pose_in_map[1] * resolution_ratio  + desdf["t"]
        else:
            pose_in_map = pose.copy()
            pose_in_map[0] = pose_in_map[0] * 10 + desdf["l"]
            pose_in_map[1] = pose_in_map[1] * 10 + desdf["t"]

        pred_poses_map.append(pose_in_map)

        #print("Before preparing metrics")

        if (evol_path is not None) & (log_extended==True):

            #### Prepare Metrics ------------------------------------------------

            ### Quality of depth prediction: Predicted depth vs. GT depth
            # Get variables
            predicted_depths = pred_depths
            gt_depths = np.array(data["gt_depth"])[t + L,:]
            # Get metrics
            metric_depth_l1_loss = F.l1_loss(torch.tensor(predicted_depths), torch.tensor(gt_depths)).item()
            metric_depth_shape_loss = F.cosine_similarity(torch.tensor(predicted_depths), torch.tensor(gt_depths), dim=-1).mean().item()

            ### Quality of matching: Matched ray vs. Predicted ray
            # Get variables
            idx_orn = int(likelihood_orn[likelihood_2d == likelihood_2d.max()][0])
            idx_x = int(likelihood_pred[1])
            idx_y = int(likelihood_pred[0])
            desdf_loc = torch.tensor(desdf["desdf"])
            V = pred_rays.shape[0]
            pad_front = V // 2
            pad_back = V - pad_front
            pad_desdf = F.pad(desdf_loc, [pad_front, pad_back], mode="circular")
            desdf_rays = pad_desdf[idx_x, idx_y, idx_orn: idx_orn + V]
            if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
                matched_rays = desdf_rays.cpu().numpy().copy()
            else:
                matched_rays = desdf_rays.cpu().numpy()[::-1].copy()  # move to CPU before converting to numpy
            predicted_rays = pred_rays.cpu().numpy()  # move pred_rays to CPU before converting to numpy
            # Get metrics
            metric_ray_l1_loss = F.l1_loss(torch.tensor(matched_rays), torch.tensor(predicted_rays)).item()


            ### Quality of localization per observation: Predicted pose vs. GT pose
            if dataset == "hge_customized_cropped":
                # Get variables
                observation_pose_in_map = np.empty(3)
                # desdf to map
                observation_pose_in_map[0] = likelihood_pred[0] * resolution_ratio + desdf["l"]
                observation_pose_in_map[1] = likelihood_pred[1] * resolution_ratio + desdf["t"]
                observation_pose_in_map[2] = likelihood_pred[2]
                # map to world
                observation_predicted_position, observation_predicted_orientation_rad = map_to_world(observation_pose_in_map[:2], observation_pose_in_map[2], affine_matrix, floorplan_correspondences)
                observation_predicted_orientation = observation_predicted_orientation_rad/np.pi*180
                gt_position, gt_orientation_rad = map_to_world(poses_map[t + L,:2], poses_map[t + L,2], affine_matrix, floorplan_correspondences)
                gt_orientation = gt_orientation_rad/np.pi*180
            elif dataset == "hge_customized_complete":
                # Get variables
                observation_pose_in_map = np.empty(3)
                # desdf to map
                observation_pose_in_map[0] = likelihood_pred[0] * resolution_ratio + desdf["l"]
                observation_pose_in_map[1] = likelihood_pred[1] * resolution_ratio + desdf["t"]
                observation_pose_in_map[2] = likelihood_pred[2]
                # map to world
                observation_predicted_position, observation_predicted_orientation_rad = map_to_world_hge_complete(observation_pose_in_map[:2], observation_pose_in_map[2], affine_matrix)
                observation_predicted_orientation = observation_predicted_orientation_rad/np.pi*180
                gt_position, gt_orientation_rad = map_to_world_hge_complete(poses_map[t + L,:2], poses_map[t + L,2], affine_matrix)
                gt_orientation = gt_orientation_rad/np.pi*180
            else:
                # Get variables
                observation_pose_in_map = np.empty(3)
                # desdf to map
                observation_pose_in_map[0] = likelihood_pred[0] * 10 + desdf["l"]
                observation_pose_in_map[1] = likelihood_pred[1] * 10 + desdf["t"]
                observation_pose_in_map[2] = likelihood_pred[2]
                # map to world
                observation_predicted_position = observation_pose_in_map[:2]*0.01
                observation_predicted_orientation = observation_pose_in_map[2]/np.pi*180
                gt_position = poses_map[t + L,:2]*0.01
                gt_orientation = poses_map[t + L,2]/np.pi*180

            # Get metrics
            metric_observation_position_err = np.linalg.norm(observation_predicted_position - gt_position)
            metric_observation_orientation_err = min((observation_predicted_orientation - gt_orientation) % 360, 360 - ((observation_predicted_orientation - gt_orientation) % 360))

            ### Quality of localization filter: Predicted posterior pose vs. GT pose
            if dataset == "hge_customized_cropped":
                # Get variables
                # map to world
                posterior_predicted_position, posterior_predicted_orientation_rad = map_to_world(pose_in_map[:2], pose_in_map[2], affine_matrix, floorplan_correspondences)
                posterior_predicted_orientation = posterior_predicted_orientation_rad/np.pi*180
                gt_position, gt_orientation_rad = map_to_world(poses_map[t + L,:2], poses_map[t + L,2], affine_matrix, floorplan_correspondences)
                gt_orientation = gt_orientation_rad/np.pi*180
            elif dataset == "hge_customized_complete":
                # Get variables
                # map to world
                posterior_predicted_position, posterior_predicted_orientation_rad = map_to_world_hge_complete(pose_in_map[:2], pose_in_map[2], affine_matrix)
                posterior_predicted_orientation = posterior_predicted_orientation_rad/np.pi*180
                gt_position, gt_orientation_rad = map_to_world_hge_complete(poses_map[t + L,:2], poses_map[t + L,2], affine_matrix)
                gt_orientation = gt_orientation_rad/np.pi*180
            else:
                # Get variables
                # map to world
                posterior_predicted_position = pose_in_map[:2]*0.01
                posterior_predicted_orientation = pose_in_map[2]/np.pi*180
                gt_position = poses_map[t + L,:2]*0.01
                gt_orientation = poses_map[t + L,2]/np.pi*180

            # Get metrics
            metric_posterior_position_err = np.linalg.norm(posterior_predicted_position - gt_position)
            metric_posterior_orientation_err = min((posterior_predicted_orientation - gt_orientation) % 360, 360 - (posterior_predicted_orientation - gt_orientation) % 360)

            ### Log metrics
            metric_depth_l1_loss_ls.append(metric_depth_l1_loss)
            metric_depth_shape_loss_ls.append(metric_depth_shape_loss)
            metric_ray_l1_loss_ls.append(metric_ray_l1_loss)
            metric_observation_position_err_ls.append(metric_observation_position_err)
            metric_observation_orientation_err_ls.append(metric_observation_orientation_err)
            metric_posterior_position_err_ls.append(metric_posterior_position_err)
            metric_posterior_orientation_err_ls.append(metric_posterior_orientation_err)


#            #### Plot Figure ------------------------------------------------
#            #print("Before preparing plots")
#            fig = plt.figure(1, figsize=(4*6, 3.4))
#            fig.clf()
#
#            ### Image
#            ax = fig.add_subplot(1, 5, 1)
#            if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
#                img_path = os.path.join(dataset_dir, scene, "rgb", '{:05d}'.format((idx_within_scene * traj_l) + (t + L)) + "-0.jpg")
#            else:
#                img_path = os.path.join(dataset_dir, scene, "rgb", '{:05d}'.format((idx_within_scene * traj_l) + (t + L)) + ".png")
#            img = mpimg.imread(img_path)
#            ax.imshow(img)
#            ax.axis('off')
#            ax.set_title("t=" + str(t))
#
#
#            # # Plot the depth image
#            # ax = fig.add_subplot(1, 6, 2)
#            # if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
#            #     img = ax.imshow(input_img_np, cmap='gray', vmin=0, vmax=40)
#            # else:
#            #     img = ax.imshow(input_img_np, cmap='gray', vmin=0, vmax=15)
#            # fig.colorbar(img, ax=ax, label='Depth [m]')
#            # ax.axis('off')
#            # #ax.set_title('Depth Image')
#
#
#            ### Quality of depth prediction: Predicted depth vs. GT depth
#            ax = fig.add_subplot(1, 5, 2)
#            ax.plot(gt_depths, label="GT Depths")
#            ax.plot(predicted_depths, label="Predicted Depths")
#            if uncertainty_bool == True:
#                ax.fill_between(range(len(predicted_depths)), predicted_depths-pred_scales, predicted_depths+pred_scales, color='C1', alpha=0.2, linewidth=0.0)
#            ax.set_title("L1 Loss: " + str(np.round(metric_depth_l1_loss, 2)) + " / Shape Loss: " + str(np.round(metric_depth_shape_loss,2)))
#            ax.legend()
#            ax.set_xlabel("Index")
#            ax.set_ylabel("Depth [m]")
#            if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
#                ax.set_ylim([0,80])  
#            else:
#                ax.set_ylim([0,10])
#
#
#            ### Quality of matching: Matched ray vs. Predicted ray
#            ax = fig.add_subplot(1, 5, 3)
#            ax.plot(matched_rays, label="Matched Rays", marker='o')
#            ax.plot(predicted_rays, label="Predicted Rays", marker='o')
#            ax.set_title("L1 Loss: " + str(np.round(metric_ray_l1_loss, 2)))
#            ax.legend()
#            ax.set_xlabel("Ray Index")
#            ax.set_ylabel("Length [m]")
#            if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
#                ax.set_ylim([0,80])  
#            else:
#                ax.set_ylim([0,10])
#
#
#            ### Quality of localization per observation: Predicted pose vs. GT pose
#            s = 0.25
#            ax = fig.add_subplot(1, 5, 4)
#            if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
#                ax.imshow(likelihood_2d, cmap="coolwarm")
#            else:
#                ax.imshow(likelihood_2d, origin="lower", cmap="coolwarm")
#            ax.set_title("Error: " + str(np.round(metric_observation_position_err,2)) + "m / " + str(np.round(metric_observation_orientation_err,2)) + "°")
#            ax.axis("off")
#            ax.quiver(
#                likelihood_pred[0],
#                likelihood_pred[1],
#                s*np.cos(likelihood_pred[2]),
#                s*np.sin(likelihood_pred[2]),
#                color="blue",
#                width=s*0.2,
#                scale_units="inches",
#                units="inches",
#                scale=1,
#                headwidth=3,
#                headlength=3,
#                headaxislength=3,
#                minlength=s*0.1,
#                angles = "xy"
#            )
#            ax.quiver(
#                gt_pose_desdf[t + L, 0],
#                gt_pose_desdf[t + L, 1],
#                s*np.cos(gt_pose_desdf[t + L, 2]),
#                s*np.sin(gt_pose_desdf[t + L, 2]),
#                color="green",
#                width=s*0.2,
#                scale_units="inches",
#                units="inches",
#                scale=1,
#                headwidth=3,
#                headlength=3,
#                headaxislength=3,
#                minlength=s*0.1,
#                angles = "xy"
#            )
#
#            ### Quality of localization per observation: Predicted pose vs. GT pose
#            ax = fig.add_subplot(1, 5, 5)
#            if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
#                ax.imshow(posterior_2d.detach().cpu().numpy(), cmap="coolwarm")
#            else:
#                ax.imshow(posterior_2d.detach().cpu().numpy(), origin="lower", cmap="coolwarm")
#            ax.set_title("Error: " + str(np.round(metric_posterior_position_err,2)) + "m / " + str(np.round(metric_posterior_orientation_err,2)) + "°")
#            ax.axis("off")
#            ax.quiver(
#                pose[0],
#                pose[1],
#                s*np.cos(pose[2]),
#                s*np.sin(pose[2]),
#                color="blue",
#                width=s*0.2,
#                scale_units="inches",
#                units="inches",
#                scale=1,
#                headwidth=3,
#                headlength=3,
#                headaxislength=3,
#                minlength=s*0.1,
#                angles = "xy"
#            )
#            ax.quiver(
#                gt_pose_desdf[t + L, 0],
#                gt_pose_desdf[t + L, 1],
#                s*np.cos(gt_pose_desdf[t + L, 2]),
#                s*np.sin(gt_pose_desdf[t + L, 2]),
#                color="green",
#                width=s*0.2,
#                scale_units="inches",
#                units="inches",
#                scale=1,
#                headwidth=3,
#                headlength=3,
#                headaxislength=3,
#                minlength=s*0.1,
#                angles = "xy"
#            )
#            plt.tight_layout()

            if not os.path.exists(
                os.path.join(evol_path, "pretty_filter_extended", str(data_idx))
            ):
                os.makedirs(os.path.join(evol_path, "pretty_filter_extended", str(data_idx)))
#            fig.savefig(
#                os.path.join(
#                    evol_path, "pretty_filter_extended", str(data_idx), str(t) + ".png"
#                )
#            )

        elif (evol_path is not None) & (log_extended==False):

            # plot posterior 2d
            fig = plt.figure(0, figsize=(20, 20))
            fig.clf()
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(
                posterior_2d.detach().cpu().numpy(), origin="lower", cmap="coolwarm"
            )
            ax.quiver(
                pose[0],
                pose[1],
                np.cos(pose[2]),
                np.sin(pose[2]),
                color="blue",
                width=0.2,
                scale_units="inches",
                units="inches",
                scale=1,
                headwidth=3,
                headlength=3,
                headaxislength=3,
                minlength=0.1,
            )
            ax.quiver(
                gt_pose_desdf[t + L, 0],
                gt_pose_desdf[t + L, 1],
                np.cos(gt_pose_desdf[t + L, 2]),
                np.sin(gt_pose_desdf[t + L, 2]),
                color="green",
                width=0.2,
                scale_units="inches",
                units="inches",
                scale=1,
                headwidth=3,
                headlength=3,
                headaxislength=3,
                minlength=0.1,
            )
            ax.axis("off")
            #ax.set_title(str(t) + " posterior")

            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(likelihood_2d, origin="lower", cmap="coolwarm")
            #ax.set_title(str(t) + " likelihood")
            ax.axis("off")
            ax.quiver(
                likelihood_pred[0],
                likelihood_pred[1],
                np.cos(likelihood_pred[2]),
                np.sin(likelihood_pred[2]),
                color="blue",
                width=0.2,
                scale_units="inches",
                units="inches",
                scale=1,
                headwidth=3,
                headlength=3,
                headaxislength=3,
                minlength=0.1,
            )
            ax.quiver(
                gt_pose_desdf[t + L, 0],
                gt_pose_desdf[t + L, 1],
                np.cos(gt_pose_desdf[t + L, 2]),
                np.sin(gt_pose_desdf[t + L, 2]),
                color="green",
                width=0.2,
                scale_units="inches",
                units="inches",
                scale=1,
                headwidth=3,
                headlength=3,
                headaxislength=3,
                minlength=0.1,
            )

            if not os.path.exists(
                os.path.join(evol_path, "pretty_filter", str(data_idx))
            ):
                os.makedirs(os.path.join(evol_path, "pretty_filter", str(data_idx)))
            fig.savefig(
                os.path.join(
                    evol_path, "pretty_filter", str(data_idx), str(t) + ".png"
                )
            )

        #print("Before transition")
        # transition
        # use ground truth to compute transitions, use relative poses
        if t + L == traj_l - 1:
            continue

        if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
            current_pose_desdf = torch.from_numpy(gt_pose_desdf[t + L, :])
            next_pose_desdf = torch.from_numpy(gt_pose_desdf[t + L + 1, :])
            #current_pose_desdf = current_pose_desdf[[1, 0, 2]]
            #next_pose_desdf = next_pose_desdf[[1, 0, 2]]
            #current_pose = current_pose_desdf
            #next_pose = next_pose_desdf
            #current_pose = current_pose_desdf*desdf_resolution
            #next_pose = next_pose_desdf*desdf_resolution
            current_pose = torch.tensor([current_pose_desdf[0] * desdf_resolution, current_pose_desdf[1] * desdf_resolution, current_pose_desdf[2]], device=current_pose_desdf.device)
            next_pose = torch.tensor([next_pose_desdf[0] * desdf_resolution, next_pose_desdf[1] * desdf_resolution, next_pose_desdf[2]], device=next_pose_desdf.device)

            transition = get_rel_pose(current_pose, next_pose)
#            prior = transit(
#                posterior, transition, sig_o=0.1, sig_x=0.1, sig_y=0.1, tsize=7, rsize=7, resolution=1
#            )
#            prior = transit(
#                posterior, transition, sig_o=0.1, sig_x=0.1, sig_y=0.1, tsize=7*200+1, rsize=7*10+1, resolution=desdf_resolution
#            )
            prior = transit(
                posterior, transition, sig_o=0.1, sig_x=0.1, sig_y=0.1, tsize=7*10+1, rsize=7*10+1, resolution=desdf_resolution
            )

            # ### State 0
            # prior = transit(
            #     posterior, transition, sig_o=0.1*10, sig_x=0.1*10, sig_y=0.1*10, tsize=7*10+1, rsize=7*10+1, resolution=desdf_resolution
            # )
        else:
            current_pose = poses[0, t + L, :]
            next_pose = poses[0, t + L + 1, :]

            transition = get_rel_pose(current_pose, next_pose)
            prior = transit(
                posterior, transition, sig_o=0.1, sig_x=0.1, sig_y=0.1, tsize=7, rsize=7
            )


        if (evol_path is not None) and (log_extended==True):
            bf_current_pose_ls.append(current_pose.cpu().numpy())
            bf_transition_ls.append(transition.cpu().numpy())


        print("transition: ", transition)
    
        end_iter = time.time()
        matching_time += matching_end - matching_start
        feature_extraction_time += feature_extraction_end - feature_extraction_start
        iteration_time += end_iter - start_iter
        n_iter += 1

    # Evaluate quality of localization per observation
    if (evol_path is not None) and (log_extended==True):
        acc_record = np.array(metric_observation_position_err_ls)
        acc_orn_record = np.array(metric_observation_orientation_err_ls)

        
        # Calculate recalls
        recall_10m = np.sum(acc_record < 10) / acc_record.shape[0]
        recall_5m = np.sum(acc_record < 5) / acc_record.shape[0]
        recall_2m = np.sum(acc_record < 2) / acc_record.shape[0]
        recall_1m = np.sum(acc_record < 1) / acc_record.shape[0]
        recall_0_5m = np.sum(acc_record < 0.5) / acc_record.shape[0]
        recall_0_1m = np.sum(acc_record < 0.1) / acc_record.shape[0]
        recall_1m_30deg = np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30)) / acc_record.shape[0]

        # Print recalls
        print("10m recall = ", recall_10m)
        print("5m recall = ", recall_5m)
        print("2m recall = ", recall_2m)
        print("1m recall = ", recall_1m)
        print("0.5m recall = ", recall_0_5m)
        print("0.1m recall = ", recall_0_1m)
        print("1m 30 deg recall = ", recall_1m_30deg)

        # Plot bar plot of recalls
        recalls = [recall_0_1m, recall_0_5m, recall_1m, recall_1m_30deg, recall_2m, recall_5m, recall_10m]
        labels = ['0.1m', '0.5m', '1m', '1m 30°', '2m', '5m', '10m']
        fig = plt.figure(3, figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(labels, recalls)
        #ax.set_xlabel('Recall Type', fontsize=14)
        ax.set_ylabel('Recall', fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylim([0, 1])
        ax.grid(True)
        plt.show()
        fig.savefig(os.path.join(evol_path, "pretty_filter_extended", str(data_idx), "recalls" + ".png"))




    if (evol_path is not None) and (log_extended==True):
        
        pred_poses_map = np.stack(pred_poses_map)

        ## Error
        error = (
            ((pred_poses_map[-(traj_l - L):, :2] - poses_map[-(traj_l - L):, :2]) ** 2).sum(axis=1)
            ** 0.5
        ) * original_resolution


        ## Bounding Box Diagonal
        # Calculate bounding box diagonal for each step
        bounding_box_diagonals = []
        for i in range(1, len(poses_map[-(traj_l - L):, :2]) + 1):
            current_positions = poses_map[-(traj_l - L):, :2][:i]
            diagonal_length = minimum_bounding_box(current_positions) * original_resolution
            bounding_box_diagonals.append(diagonal_length)
        bounding_box_diagonal = np.array(bounding_box_diagonals)


        ## Distance Travelled
        # Calculate the differences between each consecutive pair of positions
        differences = np.diff(poses_map[-(traj_l - L):, :2], axis=0)
        # Calculate the Euclidean distance for each pair
        distances = np.sqrt((differences ** 2).sum(axis=1))* original_resolution
        # Calculate the accumulated distance for each timestamp
        distance_travelled = np.concatenate(([0], np.cumsum(distances)))

        ## Plot Error vs. Distance Travelled
        fig = plt.figure(4, figsize=(6, 9))
        fig.clf()
        ax = fig.add_subplot(3, 1, 1)
        ax.plot(error)
        ax.grid()
        ax.set_xlabel("Step [-]", fontsize=14)
        ax.set_ylabel("Error [m]", fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14) 
        if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
            ax.set_ylim([0,50])  
        else:
            ax.set_ylim([0,5])
        ax = fig.add_subplot(3, 1, 2)
        ax.plot(distance_travelled, error)
        ax.grid()
        ax.set_xlabel("Distance Travelled [m]", fontsize=14)
        ax.set_ylabel("Error [m]", fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
            ax.set_ylim([0,50])  
        else:
            ax.set_ylim([0,5])
        ax = fig.add_subplot(3, 1, 3)
        ax.plot(bounding_box_diagonal, error)
        ax.grid()
        ax.set_xlabel("Bounding Box Diagonal [m]", fontsize=14)
        ax.set_ylabel("Error [m]", fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
            ax.set_ylim([0,50])
        else:
            ax.set_ylim([0,5])
        fig.tight_layout()

        if not os.path.exists(os.path.join(evol_path, "pretty_filter_extended", str(data_idx))):
            os.makedirs(os.path.join(evol_path, "pretty_filter_extended", str(data_idx)))
        fig.savefig(os.path.join(evol_path, "pretty_filter_extended", str(data_idx), "error_evolution" + ".png"))


    if (evol_path is not None) and (log_extended==True):    
            fig = plt.figure(5, figsize=(2*6, 4*3))
            fig.clf()
            ax = fig.add_subplot(4, 2, 1)
            ax.plot(metric_depth_l1_loss_ls)
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Error [m]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14) 
            if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
                ax.set_ylim([0,50])
            else:
                ax.set_ylim([0,5])
            ax.set_title("Depth Mean Absolute Error", fontsize=14)

            ax = fig.add_subplot(4, 2, 2)
            ax.plot(metric_depth_shape_loss_ls)
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Similarity [-]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_ylim([-1.05, 1.05])
            ax.set_title("Depth Cosine Similarity", fontsize=14)

            ax = fig.add_subplot(4, 2, 3)
            ax.plot(metric_ray_l1_loss_ls)
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Error [m]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
                ax.set_ylim([0,50])
            else:
                ax.set_ylim([0,5])
            ax.set_title("Ray Length Mean Abs. Error", fontsize=14)

            # Create an empty and invisible subplot at position (4, 2, 4)
            ax = fig.add_subplot(4, 2, 4)
            ax.axis('off')  # Turn off the axis for this subplot

            ax = fig.add_subplot(4, 2, 5)
            ax.plot(metric_observation_position_err_ls)
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Error [m]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
                ax.set_ylim([0,50])
            else:
                ax.set_ylim([0,5])
            ax.set_title("Observation Position Abs. Error", fontsize=14)

            ax = fig.add_subplot(4, 2, 6)
            ax.plot(metric_observation_orientation_err_ls)
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Error [°]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_ylim([0, 185])
            ax.set_title("Observation Orientation Abs. Error", fontsize=14)

            ax = fig.add_subplot(4, 2, 7)
            ax.plot(metric_posterior_position_err_ls)
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Error [m]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            if ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
                ax.set_ylim([0,50])
            else:
                ax.set_ylim([0,5])
            ax.set_title("Posterior Position Abs. Error", fontsize=14)

            ax = fig.add_subplot(4, 2, 8)
            ax.plot(metric_posterior_orientation_err_ls)
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Error [°]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_ylim([0, 185])
            ax.set_title("Posterior Orientation Abs. Error", fontsize=14)
            fig.tight_layout()

            if not os.path.exists(os.path.join(evol_path, "pretty_filter_extended", str(data_idx))):
                os.makedirs(os.path.join(evol_path, "pretty_filter_extended", str(data_idx)))
            fig.savefig(os.path.join(evol_path, "pretty_filter_extended", str(data_idx), "metrics" + ".png"))




    if (evol_path is not None) and (log_extended==True):    
            fig = plt.figure(6, figsize=(1*6, 4*3))
            fig.clf()
            ax = fig.add_subplot(6, 1, 1)
            ax.plot([arr[0] for arr in bf_current_pose_ls])
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Position [m]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14) 
            #if dataset == "hge_customized_cropped":
            #    ax.set_ylim([0,500])
            #else:
            #    ax.set_ylim([0,500])
            ax.set_title("Current Pose: 0", fontsize=14)

            ax = fig.add_subplot(6, 1, 2)
            ax.plot([arr[1] for arr in bf_current_pose_ls])
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Position [m]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_title("Current Pose: 1", fontsize=14)

            ax = fig.add_subplot(6, 1, 3)
            ax.plot([arr[2] for arr in bf_current_pose_ls])
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Orientation [rad]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            #if dataset == "hge_customized_cropped":
            #    ax.set_ylim([-2*np.pi,500])
            #else:
            #    ax.set_ylim([0,500])
            ax.set_title("Current Pose: 2", fontsize=14)

            ax = fig.add_subplot(6, 1, 4)
            ax.plot([arr[0] for arr in bf_transition_ls])
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Position [m]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            #if dataset == "hge_customized_cropped":
            #    ax.set_ylim([0,500])
            #else:
            #    ax.set_ylim([0,500])
            ax.set_title("Transition: 0", fontsize=14)

            ax = fig.add_subplot(6, 1, 5)
            ax.plot([arr[1] for arr in bf_transition_ls])
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Position [m]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            #if dataset == "hge_customized_cropped":
            #    ax.set_ylim([0,500])
            #else:
            #    ax.set_ylim([0,500])
            ax.set_title("Transition: 1", fontsize=14)

            ax = fig.add_subplot(6, 1, 6)
            ax.plot([arr[2] for arr in bf_transition_ls])
            ax.grid()
            ax.set_xlabel("Step [-]", fontsize=14)
            ax.set_ylabel("Orientation [rad]", fontsize=14)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            #if dataset == "hge_customized_cropped":
            #    ax.set_ylim([0,500])
            #else:
            #    ax.set_ylim([0,500])
            ax.set_title("Transition: 2", fontsize=14)
            fig.tight_layout()

            if not os.path.exists(os.path.join(evol_path, "pretty_filter_extended", str(data_idx))):
                os.makedirs(os.path.join(evol_path, "pretty_filter_extended", str(data_idx)))
            fig.savefig(os.path.join(evol_path, "pretty_filter_extended", str(data_idx), "bugfixing" + ".png"))



    if log_error:
        pred_poses_map = np.stack(pred_poses_map)
        # record success rate, from map to global
        last_errors = (
            ((pred_poses_map[-10:, :2] - poses_map[-10:, :2]) ** 2).sum(axis=1)
            ** 0.5
        ) * original_resolution
        # compute RMSE
        RMSE = (
            ((pred_poses_map[-10:, :2] - poses_map[-10:, :2]) ** 2)
            .sum(axis=1)
            .mean()
        ) ** 0.5 * original_resolution
        RMSEs.append(RMSE)
        print("last_errors", last_errors)
        if all(last_errors < 1):
            success_10.append(True)
        else:
            success_10.append(False)

        if all(last_errors < 0.5):
            success_5.append(True)
        else:
            success_5.append(False)

        if all(last_errors < 0.3):
            success_3.append(True)
        else:
            success_3.append(False)

        if all(last_errors < 0.2):
            success_2.append(True)
        else:
            success_2.append(False)



####
    if log_error:
        mean_depth_MAE_ls.append(np.mean(metric_depth_l1_loss_ls))
        mean_depth_cos_sim_ls.append(np.mean(metric_depth_shape_loss_ls))
        recalls_all_ls.append(np.array(recalls))
####

# In[ ]:





if log_error:
    RMSEs = np.array(RMSEs)
    success_10 = np.array(success_10)
    success_5 = np.array(success_5)
    success_3 = np.array(success_3)
    success_2 = np.array(success_2)

    print("============================================")
    print("1.0 success rate : ", success_10.sum() / len(test_set))
    print("0.5 success rate : ", success_5.sum() / len(test_set))
    print("0.3 success rate : ", success_3.sum() / len(test_set))
    print("0.2 success rate : ", success_2.sum() / len(test_set))
    print("mean RMSE succeeded : ", RMSEs[success_10].mean())
    print("mean RMSE all : ", RMSEs.mean())


# In[ ]:


if log_timing:
    feature_extraction_time = feature_extraction_time / n_iter
    matching_time = matching_time / n_iter
    iteration_time = iteration_time / n_iter

    print("============================================")
    print("feature_extraction_time : ", feature_extraction_time)
    print("matching_time : ", matching_time)
    print("iteration_time : ", iteration_time)



####
# stats II
if log_error:
    mean_mean_depth_MAE = np.mean(mean_depth_MAE_ls)
    mean_mean_depth_cos_sim = np.mean(mean_depth_cos_sim_ls)
    mean_recalls = np.mean(recalls_all_ls, axis=0)
    print("============================================")
    print("mean_mean_depth_MAE : ", mean_mean_depth_MAE)
    print("mean_mean_depth_cos_sim : ", mean_mean_depth_cos_sim)
    print("mean_recalls : ", mean_recalls)
####

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("No GPU found.")