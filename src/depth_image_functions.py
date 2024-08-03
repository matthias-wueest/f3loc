import torch
import cv2
import numpy as np
from metric3d.hubconf import *



def get_depth_image_metric3d(rgb_origin, K=np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])):

    metric3d_input_size = (616, 1064) # for vit model
    metric3d_model_name = "ViT-Large"#"ViT-Small" #
    #min_depth = 0.0
    #max_depth = 100.0

    #rgb_file = image_path
    #rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]
    #print(f"Original RGB shape: {rgb_origin.shape}")  # Debug statement

    model_name = metric3d_model_name
    intrinsic = [K[0,0], K[1,1], K[0,2], K[1,2]] #self.K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])
    
    #### adjust input size to fit pretrained model
    # keep ratio resize
    input_size = metric3d_input_size
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

    depth_image = pred_depth

    return depth_image