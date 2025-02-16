import argparse
import os

import torch
import tqdm
import yaml
from attrdict import AttrDict

from modules.comp.comp_d_net_pl import *
from modules.mono.depth_net_pl import *
from modules.mv.mv_depth_net_pl import *
from utils.data_utils import *
from utils.localization_utils import *


def evaluate_observation():
    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")

    # network to evaluate
    net_type = "d"

    # parameters
    L = 0#3  # number of the source frames
    D = 128  # number of depth planes
    d_min = 0.1  # minimum depth
    d_max = 15.0  # maximum depth
    d_hyp = -0.2  # depth transform (uniform sampling in d**d_hyp)
    #F_W = 3 / 8  # camera intrinsic, focal length / image width
    #trans_thresh = 0.005  # translation threshold (variance) if using comp_s

    F_W = 1/np.tan(0.698132)/2
    V = 8
    dv = 10


#    add_rp = (
#        False  # whether use roll and pitch angle augmentation, only used in training
#    )
#    roll = 0  # maximum roll augmentation in randian
#    pitch = 0  # maximum pitch augmentation in randian

    # paths
#    dataset_dir = os.path.join(args.dataset_path, args.dataset)
    dataset_dir = os.path.abspath("/cluster/project/cvg/data/Structured3D/Structured3D_Perspective_Full/Structured3D")
    dataset_path = os.path.abspath("/cluster/project/cvg/data/Structured3D/Structured3D_Perspective_Full/")
#    depth_dir = dataset_dir
    #log_dir = "tb_logs/my_model/version_229/checkpoints/epoch=19-step=325180.ckpt"
    desdf_path = os.path.join(dataset_path, "s3d_desdf")

    if net_type == "d":
        depth_suffix = "depth40"
    else:
        depth_suffix = "depth160"

    # instanciate dataset
#    split_file = os.path.join(dataset_dir, "split.yaml")
#    with open(split_file, "r") as f:
#        split = AttrDict(yaml.safe_load(f))
#    test_set = GridSeqDataset(
#        dataset_dir,
#        split.test,
#        L=L,
#        depth_dir=depth_dir,
#        depth_suffix=depth_suffix,
#        add_rp=add_rp,
#        roll=roll,
#        pitch=pitch,
#    )
    scene_range_test = [3250, 3499] #3260]#
#    test_set = S3DDataset(
#        dataset_dir=dataset_dir,
#        scene_range = scene_range_test,
#        depth_suffix=depth_suffix
#    )
    test_set = S3DDataset_depthanything(
        dataset_dir=dataset_dir,
        scene_range = scene_range_test,
        depth_suffix=depth_suffix
    )

    
    # create model
    #checkpoint_path="/cluster/home/wueestm/f3loc/tb_logs/my_model/version_229/checkpoints/epoch=19-step=325180.ckpt" # F3Loc own: first try
    #checkpoint_path="/cluster/home/wueestm/f3loc/tb_logs/my_model/version_238/checkpoints/epoch=49-step=406500.ckpt"
    #checkpoint_path="/cluster/home/wueestm/f3loc/tb_logs/my_model/version_237/checkpoints/epoch=15-step=130080.ckpt"
    checkpoint_path="/cluster/home/wueestm/f3loc/tb_logs/my_model/version_237/checkpoints/epoch=33-step=276420.ckpt"



    if net_type == "mvd" or net_type == "comp_s":
        mv_net = mv_depth_net_pl.load_from_checkpoint(
            checkpoint_path=os.path.join(log_dir, "mv.ckpt"),
            D=D,
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
        ).to(device)
    if net_type == "d" or net_type == "comp_s":
#        d_net = depth_net_pl.load_from_checkpoint(
#            checkpoint_path=checkpoint_path, #checkpoint_path=os.path.join(log_dir, "mono.ckpt"),
#            d_min=d_min,
#            d_max=d_max,
#            d_hyp=d_hyp,
#            D=D,
#            F_W=F_W,
#        ).to(device)
        d_net = depth_net_depthanything_pl.load_from_checkpoint(
            checkpoint_path=checkpoint_path, #checkpoint_path=os.path.join(log_dir, "mono.ckpt"),
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
            D=D,
            F_W=F_W,
        ).to(device)
    if net_type == "comp":
        comp_net = comp_d_net_pl.load_from_checkpoint(
            checkpoint_path=os.path.join(log_dir, "comp.ckpt"),
            mv_net=mv_depth_net_pl(D=D, d_hyp=d_hyp).net,
            mono_net=depth_net_pl(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D).encoder,
            L=L,
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
            D=D,
            F_W=F_W,
            use_pred=True,
        ).to(device)
        comp_net.eval()  # this is needed to disable batchnorm

    # =====================
    # ==== EVALUATION =====
    # =====================

    # get desdf for the scene
    print("load desdf ...")
    desdfs = {}
    for scene in tqdm.tqdm(test_set.scene_names):
        desdfs[scene] = np.load(
            os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True
        ).item()
        desdfs[scene]["desdf"][desdfs[scene]["desdf"] > 10] = 10  # truncate

    # get the ground truth poses
    print("load poses and maps ...")
    maps = {}
    gt_poses = {}

    # record stats extended: per trajectory
    metric_depth_l1_loss_ls = []
    metric_depth_shape_loss_ls = []

    for scene in tqdm.tqdm(test_set.scene_names):
        # load map
        occ = cv2.imread(os.path.join(dataset_dir, scene, "map.png"))[:, :, 0]
        maps[scene] = occ
        h = occ.shape[0]
        w = occ.shape[1]

#        # get poses
#        with open(os.path.join(dataset_dir, scene, "poses.txt"), "r") as f:
#            poses_txt = [line.strip() for line in f.readlines()]
#            traj_len = len(poses_txt)
#            poses = np.zeros([traj_len, 3], dtype=np.float32)
#            for state_id in range(traj_len):
#                pose = poses_txt[state_id].split(" ")
#                ## from world coordinate to map coordinate
#                #x = float(pose[0]) / 0.01 + w / 2
#                #y = float(pose[1]) / 0.01 + h / 2
#                #th = float(pose[2])
#                #poses[state_id, :] = np.array((x, y, th), dtype=np.float32)
#                # from world coordinate to map coordinate
#                x = float(pose[0]) / 0.02 + w / 2
#                y = float(pose[1]) / 0.02 + h / 2
#                th = float(pose[2])
#                poses[state_id, :] = np.array((x, y, th), dtype=np.float32)
#
#            gt_poses[scene] = poses


#### Try: 22.10.2024
        # get poses
        with open(os.path.join(dataset_dir, scene, "poses_map.txt"), "r") as f:
            poses_txt = [line.strip() for line in f.readlines()]
            traj_len = len(poses_txt)
            poses = np.zeros([traj_len, 3], dtype=np.float32)
            for state_id in range(traj_len):
                pose = poses_txt[state_id].split(" ")
                # from world coordinate to map coordinate
                x = float(pose[0])# / 0.01 + w / 2
                y = float(pose[1])# / 0.01 + h / 2
                th = float(pose[2])#
                poses[state_id, :] = np.array((x, y, th), dtype=np.float32)

            gt_poses[scene] = poses
####





    # record the accuracy
    acc_record = []
    acc_orn_record = []
    for data_idx in tqdm.tqdm(range(len(test_set))):
        data = test_set[data_idx]
        # get the scene name according to the data_idx
        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]

        # get desdf
        desdf = desdfs[scene]

        # get reference pose in map coordinate and in scene coordinate
        ref_pose_map = gt_poses[scene][idx_within_scene * (L + 1) + L, :]
        #src_pose_map = gt_poses[scene][
        #    idx_within_scene * (L + 1) : idx_within_scene * (L + 1) + L, :
        #]
        ref_pose = data["ref_pose"]
        #src_pose = data["src_pose"]

        data["ref_pose"] = torch.tensor(ref_pose, device=device).unsqueeze(0)
        #data["src_pose"] = torch.tensor(src_pose, device=device).unsqueeze(0)

#        # transform to desdf frame
#        gt_pose_desdf = ref_pose_map.copy()
#        gt_pose_desdf[0] = (gt_pose_desdf[0] - desdf["l"]) / 10
#        gt_pose_desdf[1] = (gt_pose_desdf[1] - desdf["t"]) / 10


#### Try: 22.10.2024

        # transform to desdf frame
        gt_pose_desdf = ref_pose_map.copy()
        gt_pose_desdf[0] = (gt_pose_desdf[0] - desdf["l"]) * 0.02 / 0.1
        gt_pose_desdf[1] = (gt_pose_desdf[1] - desdf["t"]) * 0.02 / 0.1

####


        # get observation
        ref_img = data["ref_img"]  # (C, H, W)
        #src_img = data["src_img"]  # (L, C, H, W)

        # get ground truth roll and pitch
        # do the gravity alignment
        # compute the attention mask
        ref_mask = None  # no masks because the dataset has zero roll pitch

        ref_img_torch = torch.tensor(ref_img, device=device).unsqueeze(0)
        ref_mask_torch = None
        data["ref_img"] = ref_img_torch
        data["ref_mask"] = ref_mask_torch

        #src_mask = None  # no masks because the dataset has zero roll pitch

        #src_img_torch = torch.tensor(src_img, device=device).unsqueeze(0)
        #data["src_img"] = src_img_torch
        #src_mask_torch = None
        #data["src_mask"] = src_mask_torch

        if net_type == "comp_s":
            # calculate the relative poses
            pose_var = (
                torch.cat((data["ref_pose"].unsqueeze(1), data["src_pose"]), dim=1)
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

        # inference
        if net_type == "mvd" or (net_type == "comp_s" and use_mv):
            data_dict = mv_net.net(data)
            pred_depths = data_dict["d"]
            pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()
        elif net_type == "d" or (net_type == "comp_s" and use_mono):
            pred_depths, attn_2d, prob = d_net.encoder(ref_img_torch, ref_mask_torch)
            pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()
        elif net_type == "comp":
            pred_dict = comp_net.comp_d_net(data)
            pred_depths = pred_dict["d_comp"].squeeze(0).detach().cpu().numpy()
#        print("data[ref_depth]: ", data["ref_depth"])
#        print("pred_depths: ", pred_depths)
        
        pred_rays = get_ray_from_depth(pred_depths, V=V, dv=dv, F_W=F_W)
        #pred_rays = get_ray_from_depth(pred_depths)
        pred_rays = torch.tensor(pred_rays, device="cpu")
#        print("pred_rays: ", pred_rays)


        # localize with the desdf using the prediction
        prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = localize(
            torch.tensor(desdf["desdf"]), pred_rays
        )

        # calculate accuracy
        acc = np.linalg.norm(pose_pred[:2] - gt_pose_desdf[:2], 2.0) * 0.1
        acc_record.append(acc)
        acc_orn = (pose_pred[2] - gt_pose_desdf[2]) % (2 * np.pi)
        acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180
        acc_orn_record.append(acc_orn)


        
        # Get metrics
        predicted_depths = pred_depths
        gt_depths = np.array(data["ref_depth"])
        metric_depth_l1_loss = F.l1_loss(torch.tensor(predicted_depths), torch.tensor(gt_depths)).item()
        metric_depth_shape_loss = F.cosine_similarity(torch.tensor(predicted_depths), torch.tensor(gt_depths), dim=-1).mean().item()

        ### Log metrics
        metric_depth_l1_loss_ls.append(metric_depth_l1_loss)
        metric_depth_shape_loss_ls.append(metric_depth_shape_loss)
        #metric_ray_l1_loss_ls.append(metric_ray_l1_loss)
        #metric_observation_position_err_ls.append(metric_observation_position_err)
        #metric_observation_orientation_err_ls.append(metric_observation_orientation_err)
        #metric_posterior_position_err_ls.append(metric_posterior_position_err)
        #metric_posterior_orientation_err_ls.append(metric_posterior_orientation_err)
        #gt_depths_ls.append(gt_depths)
        #predicted_depths_ls.append(predicted_depths)




#    if log_error:
#        mean_depth_MAE_ls.append(np.mean(metric_depth_l1_loss_ls))
#        mean_depth_cos_sim_ls.append(np.mean(metric_depth_shape_loss_ls))
#        mean_ray_MAE_ls.append(np.mean(metric_ray_l1_loss_ls))
#        mean_obs_pos_err_ls.append(np.mean(metric_observation_position_err_ls))
#        mean_obs_orn_err_ls.append(np.mean(metric_observation_orientation_err_ls))
#        mean_post_pos_err_ls.append(np.mean(metric_posterior_position_err_ls))
#        mean_post_orn_err_ls.append(np.mean(metric_posterior_orientation_err_ls))
#        recalls_all_ls.append(np.array(recalls))
#        successes_all_ls.append(np.array(successes))
#        RMSEs_all_ls.append(np.array(RMSE))
#
#        metric_depth_l1_loss_ls_ls.append(metric_depth_l1_loss_ls)
#        metric_depth_shape_loss_ls_ls.append(metric_depth_shape_loss_ls)
#        metric_ray_l1_loss_ls_ls.append(metric_ray_l1_loss_ls)
#        metric_observation_position_err_ls_ls.append(metric_observation_position_err_ls)
#        metric_observation_orientation_err_ls_ls.append(metric_observation_orientation_err_ls)
#        metric_posterior_position_err_ls_ls.append(metric_posterior_position_err_ls)
#        metric_posterior_orientation_err_ls_ls.append(metric_posterior_orientation_err_ls)
#        gt_depths_ls.append(gt_depths)
#        predicted_depths_ls.append(predicted_depths)




    acc_record = np.array(acc_record)
    acc_orn_record = np.array(acc_orn_record)
    print("10m recall = ", np.sum(acc_record < 10) / acc_record.shape[0])
    print("5m recall = ", np.sum(acc_record < 5) / acc_record.shape[0])
    print("2m recall = ", np.sum(acc_record < 2) / acc_record.shape[0])
    print("1m recall = ", np.sum(acc_record < 1) / acc_record.shape[0])
    print("0.5m recall = ", np.sum(acc_record < 0.5) / acc_record.shape[0])
    print("0.1m recall = ", np.sum(acc_record < 0.1) / acc_record.shape[0])
    print(
        "1m 30 deg recall = ",
        np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30))
        / acc_record.shape[0],
    )



    # stats II
    if True:

        mean_mean_depth_MAE = np.mean(metric_depth_l1_loss_ls)
        mean_mean_depth_cos_sim = np.mean(metric_depth_shape_loss_ls)
    #    mean_mean_depth_MAE = np.mean(mean_depth_MAE_ls)
    #    mean_mean_depth_cos_sim = np.mean(mean_depth_cos_sim_ls)
    #    print("============================================")
        print("mean_mean_depth_MAE : ", mean_mean_depth_MAE)
        print("mean_mean_depth_cos_sim : ", mean_mean_depth_cos_sim)





if __name__ == "__main__":
    evaluate_observation()