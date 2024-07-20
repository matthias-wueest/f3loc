import lightning.pytorch as pl
import torch.optim as optim

from .depth_net import *


class depth_net_pl(pl.LightningModule):
    """
    lightning wrapper for the depth_net
    """

    def __init__(
        self,
        shape_loss_weight=None,
        lr=1e-3,
        d_min=0.1,
        d_max=15.0,
        d_hyp=-0.2,
        D=128,
        F_W=3 / 8,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.F_W = F_W
        self.encoder = depth_net(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step_original(self, batch, batch_idx):
        # train the ray depths
        rays, attn_2d, _ = self.encoder(
            batch["img"], batch["mask"] if "mask" in batch else None
        )
        loss = F.l1_loss(rays, batch["gt_rays"])
        self.log("l1_loss-train", loss)
 
        if self.shape_loss_weight is not None:
            shape_loss = self.shape_loss_weight * (
                1 - F.cosine_similarity(rays, batch["gt_rays"]).mean()
            )
            loss += shape_loss
            self.log("shape_loss-train", shape_loss)
 
        self.log("loss-train", loss)
        return loss
 
    def validation_step_original(self, batch, batch_idx):
        # train the ray distance
        rays, attn_2d, prob = self.encoder(
            batch["img"], batch["mask"] if "mask" in batch else None
        )
        loss = F.l1_loss(rays, batch["gt_rays"])
        self.log("l1_loss-valid", loss)
        if self.shape_loss_weight is not None:
            shape_loss = 1 - F.cosine_similarity(rays, batch["gt_rays"]).mean()
            loss += shape_loss
            self.log("shape_loss-valid", shape_loss)
 
        self.log("loss-valid", loss)

    def training_step_own_original(self, batch, batch_idx):
        
        d_mono, _, prob_mono = self.encoder(
            batch["ref_img"], batch["ref_mask"]
        )  # d_mono: (N, fWm), prob_mono: (N, fWm, D)
        #prob_mono = prob_mono.unsqueeze(1)  # (N, 1, fWm, D)
        #d_mono = d_mono.unsqueeze(1)  # (N, 1, fWm)
        #mono_dict = {"d_mono": d_mono, "prob_mono": prob_mono}
        l1_loss = F.l1_loss(d_mono, batch["ref_depth"])
        self.log("l1_loss-train", l1_loss, prog_bar=True, logger=True)

        loss = l1_loss

        if self.shape_loss_weight is not None:
            shape_loss = self.shape_loss_weight * (
                1 - F.cosine_similarity(d_mono, batch["ref_depth"], dim=-1).mean()
            )
            self.log("shape_loss-train", shape_loss, prog_bar=True, logger=True)

            loss += shape_loss

        self.log("loss-train", loss, prog_bar=True, logger=True)
        return loss



    def training_step(self, batch, batch_idx):
        d_mono, _, prob_mono = self.encoder(
            batch["ref_img"], batch["ref_mask"]
        )  # d_mono: (N, fWm), prob_mono: (N, fWm, D)

        # Check for NaNs in d_mono and batch["ref_depth"]
        if torch.isnan(d_mono).any() or torch.isnan(batch["ref_depth"]).any():
            print(f"NaNs found in d_mono or batch['ref_depth'] at batch_idx {batch_idx}")
            print("d_mono.shape: ", d_mono.shape)
            print("batch[ref_depth].shape : ", batch["ref_depth"].shape)
            return None

        l1_loss = F.l1_loss(d_mono, batch["ref_depth"])
        self.log("l1_loss-train", l1_loss, prog_bar=True, logger=True)

        loss = l1_loss

        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(d_mono, batch["ref_depth"], dim=-1).mean()
            
            # Check for NaNs in cosine_sim
            if torch.isnan(cosine_sim).any():
                print(f"NaNs found in cosine_similarity at batch_idx {batch_idx}")
                return None

            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            self.log("shape_loss-train", shape_loss, prog_bar=True, logger=True)

            loss += shape_loss

        self.log("loss-train", loss, prog_bar=True, logger=True)
        return loss




    def validation_step(self, batch, batch_idx):
        d_mono, _, prob_mono = self.encoder(
            batch["ref_img"], batch["ref_mask"]
        )  # d_mono: (N, fWm), prob_mono: (N, fWm, D)
        #prob_mono = prob_mono.unsqueeze(1)  # (N, 1, fWm, D)
        #d_mono = d_mono.unsqueeze(1)  # (N, 1, fWm)
        l1_loss = F.l1_loss(d_mono, batch["ref_depth"])
        self.log("l1_loss-valid", l1_loss, prog_bar=True, logger=True, on_epoch=True)

        loss = l1_loss

        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(d_mono, batch["ref_depth"], dim=-1).mean()
            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            self.log("shape_loss-valid", shape_loss, prog_bar=True, logger=True, on_epoch=True)

            loss += shape_loss

        self.log("loss-valid", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss