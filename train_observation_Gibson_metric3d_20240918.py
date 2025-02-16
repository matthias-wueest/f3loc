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


def train_observation():
    net_type = "d"#"comp"#
    dataset = "gibson_f"
    #dataset_path = "/cluster/scratch/wueestm/gibson/Gibson_Floorplan_Localization_Dataset"
    dataset_path= "/cluster/project/cvg/data/gibson/Gibson_Floorplan_Localization_Dataset"

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")


    # parameters
    L = 3  # number of the source frames
    D = 128  # number of depth planes
    d_min = 0.1  # minimum depth
    d_max = 15.0  # maximum depth
    d_hyp = -0.2  # depth transform (uniform sampling in d**d_hyp)
    F_W = 3 / 8  # camera intrinsic, focal length / image width
    trans_thresh = 0.005  # translation threshold (variance) if using comp_s

    add_rp = (
        True  # whether use roll and pitch angle augmentation, only used in training
    )
    roll = 0.1  # maximum roll augmentation in radian
    pitch = 0.1  # maximum pitch augmentation in radian


    # paths
    dataset_dir = os.path.join(dataset_path, dataset)
    depth_dir = dataset_dir
    #log_dir = ckpt_path
    desdf_path = os.path.join(dataset_path, "desdf")

    if net_type == "d":
        depth_suffix = "depth40"
    else:
        depth_suffix = "depth160"

    # instantiate dataset
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    # Define data
    train_dataset = GridSeqDataset_Gibson_metric3d(
        dataset_dir,
        split.train,
        L=L,
        depth_dir=depth_dir,
        depth_suffix=depth_suffix,
        add_rp=add_rp,
        roll=roll,
        pitch=pitch,
    )

    val_dataset = GridSeqDataset_Gibson_metric3d(
        dataset_dir,
        split.val,
        L=L,
        depth_dir=depth_dir,
        depth_suffix=depth_suffix,
        add_rp=add_rp,
        roll=roll,
        pitch=pitch,
    )


#   # Define data
#    train_dataset = GridSeqDataset(
#        dataset_dir,
#        split.train,
#        L=L,
#        depth_dir=depth_dir,
#        depth_suffix=depth_suffix,
#        add_rp=add_rp,
#        roll=roll,
#        pitch=pitch,
#    )
#
#    val_dataset = GridSeqDataset(
#        dataset_dir,
#        split.val,
#        L=L,
#        depth_dir=depth_dir,
#        depth_suffix=depth_suffix,
#        add_rp=add_rp,
#        roll=roll,
#        pitch=pitch,
#    )


    #dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    #dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)


    # Define model 
    model = depth_net_metric3d_pl(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D)
    #model = depth_net_uncertainty_pl(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D)
    #model = depth_net_pl(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D)
    #model = comp_d_net_pl(mv_net=mv_depth_net_pl(D=D, d_hyp=d_hyp).net,
    #            mono_net=depth_net_pl(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D).encoder,
    #            L=L,
    #            d_min=d_min,
    #            d_max=d_max,
    #            d_hyp=d_hyp,
    #            D=D,
    #            F_W=F_W,
    #            use_pred=True).to(device)

    #from lightning.pytorch.callbacks import ModelCheckpoint
    #checkpoint_callback = ModelCheckpoint(
    #dirpath='checkpoints',  # Directory where the checkpoints will be saved
    #filename='best-checkpoint',  # Filename for the checkpoint
    #save_top_k=1,  # Save only the best model
    #monitor='val_loss',  # Metric to monitor to decide the best model
    #mode='min'  # Minimize the monitored metric
        #)



    from lightning.pytorch.loggers import TensorBoardLogger  # Import TensorBoardLogger

    # Setup TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="my_model")


    # Train the model
    #trainer = Lightning.Trainer(max_steps=10, max_epochs=1, enable_checkpointing=True)
    trainer = Lightning.Trainer(max_epochs=20, enable_checkpointing=True, logger=logger)
    #trainer = Lightning.Trainer(max_epochs=50, enable_checkpointing=True, logger=logger)
    #trainer = Lightning.Trainer(max_epochs=100, enable_checkpointing=True, logger=logger)
    #trainer = Lightning.Trainer(max_steps=8, max_epochs=1, enable_checkpointing=True)
    #trainer = Lightning.Trainer(max_epochs=8, enable_checkpointing=True)
    #trainer = Lightning.Trainer(fast_dev_run=10, enable_checkpointing=True)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)



if __name__ == "__main__":
    train_observation()
