#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import os
from pathlib import Path
import argparse

import torch
import tqdm
import yaml
from attrdict import AttrDict

from modules.comp.comp_d_net_pl import *
from modules.mono.depth_net_pl import *
from modules.mv.mv_depth_net_pl import *
from utils.data_utils import *
from utils.localization_utils import *


from torch.utils.data import DataLoader
import lightning as Lightning
from lightning.pytorch import seed_everything

# In[2]:


net_type = "d"#"comp"#

# dataset = "gibson_f"
# #dataset_path = "/cluster/scratch/wueestm/gibson/Gibson_Floorplan_Localization_Dataset"
# dataset_path= "/cluster/project/cvg/data/gibson/Gibson_Floorplan_Localization_Dataset"

dataset = "hge_customized_complete" #"hge_customized_cropped"
dataset_path = "/cluster/project/cvg/data/lamar/HGE_customized_complete" #"/cluster/project/cvg/data/lamar/HGE_customized_cropped"


# In[3]:


# get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("======= USING DEVICE : ", device, " =======")


# In[4]:


# # parameters
# L = 0  # number of the source frames
# D = 128  # number of depth planes
# d_min = 0.1  # minimum depth
# d_max = 15.0  # maximum depth
# d_hyp = -0.2  # depth transform (uniform sampling in d**d_hyp)
# #F_W = 3 / 8  # camera intrinsic, focal length / image width
# #trans_thresh = 0.005  # translation threshold (variance) if using comp_s


L = 0  # number of the source frames
D = 128  # number of depth planes
d_min = 0.1  # minimum depth
d_max = 100  # maximum depth
d_hyp = 0.2  # depth transform (uniform sampling in d**d_hyp)
shape_loss_weight=None #40 #None #1
lr= 1e-3 #1e-3 #1e-1 #
shuffle_train = True
enable_checkpointing = True
batch_size = 4
max_epochs = 20#50# 
seed = 42

# In[5]:


add_rp = (
    False  # whether use roll and pitch angle augmentation, only used in training
)
roll = 0.00001  # maximum roll augmentation in radian
pitch = 0.00001  # maximum pitch augmentation in radian


# In[6]:


# paths
dataset_dir = os.path.join(dataset_path, dataset)
depth_dir = dataset_dir
#log_dir = ckpt_path
desdf_path = os.path.join(dataset_path, "desdf")

if (net_type == "d") & ((dataset == "hge_customized_cropped") or (dataset =="hge_customized_complete")):
    depth_suffix = "depth90"
elif net_type == "d":
    depth_suffix = "depth40"
else:
    depth_suffix = "depth160"


# In[7]:


# instantiate dataset
split_file = os.path.join(dataset_dir, "split.yaml")
with open(split_file, "r") as f:
    split = AttrDict(yaml.safe_load(f))


# In[8]:

# Set seed  
seed_everything(seed, workers=True)

# # Define data
# train_dataset = GridSeqDataset_hge_customized_cropped_gravity_align(
#     dataset_dir,
#     split.train,
#     L=L,
#     depth_dir=depth_dir,
#     depth_suffix=depth_suffix,
#     add_rp=add_rp,
#     roll=roll,
#     pitch=pitch,
# )
# 
# 
# val_dataset = GridSeqDataset_hge_customized_cropped_gravity_align(
#     dataset_dir,
#     split.val,
#     L=L,
#     depth_dir=depth_dir,
#     depth_suffix=depth_suffix,
#     add_rp=add_rp,
#     roll=roll,
#     pitch=pitch,
# )

# Define data
train_dataset = GridSeqDataset_hge_customized_metric3d_depths_normals_segmentation(
    dataset_dir,
    split.train,
    L=L,
    depth_dir=depth_dir,
    depth_suffix=depth_suffix,
    add_rp=add_rp,
    roll=roll,
    pitch=pitch,
)


val_dataset = GridSeqDataset_hge_customized_metric3d_depths_normals_segmentation(
    dataset_dir,
    split.val,
    L=L,
    depth_dir=depth_dir,
    depth_suffix=depth_suffix,
    add_rp=add_rp,
    roll=roll,
    pitch=pitch,
)

# In[9]:



#train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
#train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)

# In[10]:


#val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# In[11]:


# Define model 
#model = depth_net_pl(shape_loss_weight=shape_loss_weight, lr=lr, d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D)
#model = depth_net_metric3d_pl(shape_loss_weight=shape_loss_weight, lr=lr, d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D)
#model = depth_net_metric3d_depths_normals_pl(shape_loss_weight=shape_loss_weight, lr=lr, d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D)
model = depth_net_metric3d_depths_normals_segmentation_pl(shape_loss_weight=shape_loss_weight, lr=lr, d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D)

# In[12]:


from lightning.pytorch.loggers import TensorBoardLogger  # Import TensorBoardLogger

# Setup TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="my_model")


# In[13]:


# Train the model
# Set seed  
seed_everything(seed, workers=True)
#trainer = Lightning.Trainer(max_steps=10, max_epochs=1, enable_checkpointing=True)
#trainer = Lightning.Trainer(max_epochs=100, enable_checkpointing=True, logger=logger)
trainer = Lightning.Trainer(max_epochs=max_epochs, enable_checkpointing=enable_checkpointing, logger=logger)
#trainer = Lightning.Trainer(max_steps=8, max_epochs=1, enable_checkpointing=True)
#trainer = Lightning.Trainer(max_epochs=8, enable_checkpointing=True)
#trainer = Lightning.Trainer(fast_dev_run=10, enable_checkpointing=True)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




