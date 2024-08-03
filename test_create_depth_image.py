

from metric3d.hubconf import *



if __name__ == '__main__':
    print("before first imports")
    import cv2
    import numpy as np
    #### prepare data
    #rgb_file = 'data/kitti_demo/rgb/0000000050.png'
    #depth_file = 'data/kitti_demo/depth/0000000050.png'
    #intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
    #gt_depth_scale = 256.0
    print("after first imports")
    #model_name_ls = ["ViT-Small", "ViT-Large", "ViT-giant2"]
    #rgb_filename_ls = ['00000-0', '00090-0', '00120-0']
    model_name_ls = ["ViT-Large"]
    rgb_filename_ls = ['00000-0', '00090-0', '00120-0']
    #rgb_filename_ls = ['aligned_image_0', 'aligned_image_90', 'aligned_image_120']
    
    for model_name in model_name_ls:
        for rgb_filename in rgb_filename_ls:

            rgb_file = 'metric3d/data/hge_customized_complete/non-aligned/rgb/' + rgb_filename + ".png"
            depth_file = None
            intrinsic = [1596, 1596, 960, 720] #[240, 240, 1440/2, 1920/2] # [240, 240, 320, 240]
            rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]

            #### adjust input size to fit pretrained model
            # keep ratio resize
            input_size = (616, 1064) # for vit model
            # input_size = (544, 1216) # for convnext model
            h, w = rgb_origin.shape[:2]
            scale = min(input_size[0] / h, input_size[1] / w)
            rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
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
            print("after padding of input size")
            #### normalize
            mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
            std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
            rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
            rgb = torch.div((rgb - mean), std)
            rgb = rgb[None, :, :, :].cuda()
            print("after normalize")
            ###################### canonical camera space ######################
            # inference
            if model_name == "ViT-Small":
                model = metric3d_vit_small(pretrain=True)
            elif model_name == "ViT-Large":
                model = metric3d_vit_large(pretrain=True)
            elif model_name == "ViT-giant2":
                model = metric3d_vit_giant2(pretrain=True)
            print("after loading of model")
            model.cuda().eval()
            start_time = time.time()
            with torch.no_grad():
                pred_depth, confidence, output_dict = model.inference({'input': rgb})
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")

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

            #### you can now do anything with the metric depth 
            # such as evaluate predicted depth
            if depth_file is not None:
                gt_depth = cv2.imread(depth_file, -1)
                gt_depth = gt_depth / gt_depth_scale
                gt_depth = torch.from_numpy(gt_depth).float().cuda()
                assert gt_depth.shape == pred_depth.shape
                
                mask = (gt_depth > 1e-8)
                abs_rel_err = (torch.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
                print('abs_rel_err:', abs_rel_err.item())

            #### normal are also available
            if 'prediction_normal' in output_dict: # only available for Metric3Dv2, i.e. vit model
                pred_normal = output_dict['prediction_normal'][:, :3, :, :]
                normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
                # un pad and resize to some size if needed
                pred_normal = pred_normal.squeeze()
                pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
                # you can now do anything with the normal
                # such as visualize pred_normal
                pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
                pred_normal_vis = (pred_normal_vis + 1) / 2
                cv2.imwrite('normal_vis.png', (pred_normal_vis * 255).astype(np.uint8))



            ### Plot
            # Assuming pred_depth is your tensor with the predicted depth map
            # and rgb_file is the path to your RGB image

            # Convert the predicted depth tensor to a NumPy array
            pred_depth_np = pred_depth.cpu().numpy()

            # Prepare to save the plot
            # Get the directory of the RGB image
            rgb_dir = os.path.dirname(rgb_file)

            # Get the directory above the current directory
            save_dir = os.path.dirname(rgb_dir)

            # Define the path for the predicted depth image
            pred_depth_folder = os.path.join(save_dir, 'pred_depth', model_name)
            os.makedirs(pred_depth_folder, exist_ok=True)

            # Define the path for saving the depth map image
            save_path_image = os.path.join(pred_depth_folder, "pred_depth_" + rgb_filename + ".png")
            save_path_array = os.path.join(pred_depth_folder, "pred_depth_" + rgb_filename + ".npy")

            # Plot and save the depth map
            plt.figure(figsize=(8, 8))
            #plt.imshow(pred_depth_np, cmap='gray', vmin=np.min(pred_depth_np), vmax=np.max(pred_depth_np))  # Use 'gray' colormap
            plt.imshow(pred_depth_np, cmap='gray', vmin=0, vmax=40)  # Use 'gray' colormap
            #plt.colorbar(label='Depth')
            #plt.title('Predicted Depth Map')
            #plt.colorbar()
            plt.axis('off')  # Hide the axes
            plt.savefig(save_path_image, bbox_inches='tight', pad_inches=0, dpi=300)  # Save the image with high resolution
            np.save(save_path_array, pred_depth_np)
            plt.close()  # Close the plot to free up memory

            print(f"Depth map saved to {save_path_image}")

            print(pred_depth_np)
            print(pred_depth_np.max())
    
    #np.save(pred_depth_folder + "/pred_depth.npy", pred_depth_np)
    #pred_depth_np_loaded = np.load(pred_depth_folder + "/pred_depth.npy")
    #print("pred_depth_np_loaded.shape: ")
    #print(pred_depth_np_loaded.shape)