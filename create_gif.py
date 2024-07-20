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


# In[ ]:







# In[2]:


net_type = "d"
#dataset = "gibson_t" # "gibson_g" # 
dataset = "hge_customized_cropped"
#dataset = "hge_customized_complete"

if dataset == "gibson_t":
    dataset_path = "/cluster/project/cvg/data/gibson/Gibson_Floorplan_Localization_Dataset"
    #evol_path = "./evol_path/gibson_f/gt" #evol_path = "./evol_path/gibson_f/mono"
    evol_path = "./evol_path/gibson_f/" #evol_path = "./evol_path/gibson_f/mono"
    desdf_resolution = 0.1
    orn_slice = 36
elif dataset == "hge_customized_cropped":
    dataset_path = "/cluster/project/cvg/data/lamar/HGE_customized_cropped"
    evol_path = "./evol_path/hge_customized_cropped/gt"
    desdf_resolution = 0.1
    orn_slice = 36
elif dataset =="hge_customized_complete":
    dataset_path = "/cluster/project/cvg/data/lamar/HGE_customized_complete"
    evol_path = "./evol_path/hge_customized_complete/gt"
    desdf_resolution = 0.1
    orn_slice = 36   

ckpt_path = "./logs"
traj_len = 100#8#100#100#50


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
    test_set = TrajDataset_hge_customized_cropped(
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

else:
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








import imageio
import os
import tqdm
from PIL import Image
import numpy as np

def extract_number(filename):
    try:
        return int(filename.split('.')[0])
    except (IndexError, ValueError):
        return -1  # Return a default value in case of error

def resize_image(image, size):
    return image.resize(size, Image.Resampling.LANCZOS)

#evol_path = "./evol_path/gibson_f/mono"

#evol_path = "/cluster/home/wueestm/f3loc/evol_path/hge_customized_cropped/gt/pretty_filter_extended/0"
#evol_path = "/cluster/home/wueestm/f3loc/evol_path/hge_customized_cropped/gt/"#pretty_filter_extended/0"
#evol_path = "/cluster/home/wueestm/f3loc/evol_path/hge_customized_complete/gt/"#pretty_filter_extended/0"
evol_path = "/cluster/home/wueestm/f3loc/evol_path/hge_customized_cropped/gt/"


### Create GIF
# for data_idx in tqdm.tqdm(range(0,1)):
# for data_idx in tqdm.tqdm(range(1,len(test_set))):
for data_idx in tqdm.tqdm(range(0,len(test_set))):

    image_folder = os.path.join(evol_path, "pretty_filter_extended", str(data_idx))
    gif_path = os.path.join(image_folder, 'animation.gif')  # Specify the full path for the output GIF file

    # List of image filenames, sorted by the number in their name
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".png") and extract_number(img) != -1]

    # Debug: Print the image filenames to inspect their format
    print("Image files found:", image_files)

    # Sort the images based on the extracted number
    images = sorted(image_files, key=extract_number)

    # Read images into a list
    image_list = []
    target_size = None
    for filename in images:
        image_path = os.path.join(image_folder, filename)
        try:
            image = Image.open(image_path)
            if target_size is None:
                target_size = image.size  # Set the target size to the size of the first image
                print(f"Target size set to: {target_size}")
            image = resize_image(image, target_size)
            image_list.append(np.array(image))
            print(f"Loaded image: {filename}")
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")

    # Check if image_list is empty
    if not image_list:
        print("No valid images were loaded. Exiting.")
        continue

    # Create a GIF from the list of images
    imageio.mimsave(gif_path, image_list, duration=1000)  # Adjust duration as needed

    print(f"GIF saved to {gif_path}")