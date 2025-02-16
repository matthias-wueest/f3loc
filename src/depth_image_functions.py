import torch
import torch.nn.functional as F
import cv2
import numpy as np
from metric3d.hubconf import *
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image



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




def get_depth_image_metric3d_small(rgb_origin, K=np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])):

    metric3d_input_size = (616, 1064) # for vit model
    metric3d_model_name = "ViT-Small" #"ViT-Large"#
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


def get_depth_image_metric3d_normals(rgb_origin, K=np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])):

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


    #### normal
    pred_normal = output_dict['prediction_normal'][:, :3, :, :]
    #normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
    
    # un pad
    pred_normal = pred_normal.squeeze()
    pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]

    # upsample to original size
    pred_normal = torch.nn.functional.interpolate(pred_normal[None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
    normal_image = pred_normal


    return depth_image, normal_image



def get_segmentation_image_try(rgb_origin, K=np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])):
    # Load the image processor and model in half precision
    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").half()

    # Preprocess the image and forward it through the model
    inputs = image_processor(images=rgb_origin, return_tensors="pt").to(torch.float16)
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(0)  # Remove batch dimension, shape (num_labels, height/4, width/4)

    # Get the dimensions of the original image
    original_height, original_width = rgb_origin.shape[:2]

    # Resize the logits to the original image size using half precision
    logits_resized = torch.nn.functional.interpolate(
        logits.unsqueeze(0),  # Add batch dimension back for interpolation
        size=(original_height, original_width),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).cpu()  # Remove batch dimension and move to CPU

    # Convert the resized logits to a NumPy array
    segmentation_image = logits_resized.numpy()

    return segmentation_image


def get_segmentation_image(rgb_origin, K=np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]])):

    # Load the image processor and model
    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    
    # Preprocess the image and forward it through the model
    inputs = image_processor(images=rgb_origin, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # Get the logits for the first image in the batch
    logits = logits.squeeze(0)  # Remove batch dimension, shape (num_labels, height/4, width/4)

    # Get the dimensions of the original image
    original_height, original_width = rgb_origin.shape[:2]

    # print("original_width: ", original_width)
    # print("original_height: ", original_height)
    # print("logits.shape: ", logits.shape)
    # print("logits.shape[1]: ", logits.shape[1])
    # print("logits.shape[2]: ", logits.shape[2])

    # Resize the logits to the original image size
    #resize_factor = (original_height // logits.shape[1], original_width // logits.shape[2])
    logits_resized = torch.nn.functional.interpolate(
    logits.unsqueeze(0),  # Add batch dimension back for interpolation
    size=(original_height, original_width),
    mode='bilinear',
    align_corners=False
    ).squeeze(0)  # Remove batch dimension

    # Convert the resized logits to a NumPy array
    logits_np = logits_resized.detach().cpu().numpy()
    #print("logits_np.shape: ", logits_np.shape)
    segmentation_image = logits_np

    return segmentation_image


def softmax_pytorch(logits, mask=None, device=None):
    """
    Apply softmax to the logits to get probabilities using PyTorch.
    
    Input:
        logits: input segmentation map of shape (N, H, W), where N is the number of channels.
        mask: optional mask indicating which pixels should be set to 0 in the output probabilities.
        device: the device to perform the computation on ('cpu' or 'cuda').
    
    Output:
        probabilities: output probabilities of shape (N, H, W), where N is the number of channels.
    """

    # Choose device: 'cuda' if available, else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    # Convert logits to PyTorch tensor and move to device
    logits = torch.tensor(logits, dtype=torch.float32).to(device)
    
    # Apply softmax along the channel dimension (dim=0)
    probabilities = F.softmax(logits, dim=0)
    
    # Apply mask if provided
    if mask is not None:
        mask = torch.tensor(mask, dtype=torch.float32).to(device)
        probabilities[:, mask == 0] = 0
    
    return probabilities.cpu().numpy()