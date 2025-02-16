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



class depth_net_metric3d_pl(pl.LightningModule):
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
        self.encoder = depth_net_metric3d(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

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



class depth_net_metric3d_uncertainty_pl(pl.LightningModule):
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
        self.encoder = depth_net_metric3d_uncertainty(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def laplace_nll_loss(self, y_true, loc, scale):
        """Compute the negative log-likelihood loss for Laplace distribution."""
        return torch.mean(torch.abs(y_true - loc) / scale + torch.log(2 * scale))

    def training_step(self, batch, batch_idx):
        # Predict location (mu) and scale (b) using the modified encoder
        loc, scale, _, prob_mono = self.encoder(batch["ref_img"], batch["ref_mask"])  # loc: (N, fW), scale: (N, fW)

        # Check for NaNs in loc, scale, and batch["ref_depth"]
        if torch.isnan(loc).any() or torch.isnan(scale).any() or torch.isnan(batch["ref_depth"]).any():
            print(f"NaNs found in loc, scale, or batch['ref_depth'] at batch_idx {batch_idx}")
            return None

        # Compute Laplace NLL loss using the predicted location and scale
        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        self.log("nll_loss-train", nll_loss, prog_bar=True, logger=True)

        loss = nll_loss

        # Optionally add shape loss based on cosine similarity
        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            
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
        # Predict location (mu) and scale (b) using the modified encoder
        loc, scale, _, prob_mono = self.encoder(batch["ref_img"], batch["ref_mask"])  # loc: (N, fW), scale: (N, fW)

        # Compute Laplace NLL loss using the predicted location and scale
        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        self.log("nll_loss-valid", nll_loss, prog_bar=True, logger=True, on_epoch=True)

        loss = nll_loss

        # Optionally add shape loss based on cosine similarity
        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            self.log("shape_loss-valid", shape_loss, prog_bar=True, logger=True, on_epoch=True)

            loss += shape_loss

        self.log("loss-valid", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss


    


class depth_net_metric3d_depths_normals_pl(pl.LightningModule):
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
        self.encoder = depth_net_metric3d_depths_normals(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

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







class depth_net_metric3d_depths_normals_segmentation_pl(pl.LightningModule):
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
        self.encoder = depth_net_metric3d_depths_normals_segmentation(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

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




class depth_net_uncertainty_pl(pl.LightningModule):
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
        self.encoder = depth_net_uncertainty(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def laplace_nll_loss(self, y_true, loc, scale):
        """Compute the negative log-likelihood loss for Laplace distribution."""
        return torch.mean(torch.abs(y_true - loc) / scale + torch.log(2 * scale))

    def training_step(self, batch, batch_idx):
        # Predict location (mu) and scale (b) using the modified encoder
        loc, scale, _, prob_mono = self.encoder(batch["ref_img"], batch["ref_mask"])  # loc: (N, fW), scale: (N, fW)

        # Check for NaNs in loc, scale, and batch["ref_depth"]
        if torch.isnan(loc).any() or torch.isnan(scale).any() or torch.isnan(batch["ref_depth"]).any():
            print(f"NaNs found in loc, scale, or batch['ref_depth'] at batch_idx {batch_idx}")
            return None

        # Compute Laplace NLL loss using the predicted location and scale
        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        self.log("nll_loss-train", nll_loss, prog_bar=True, logger=True)

        loss = nll_loss

        # Optionally add shape loss based on cosine similarity
        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            
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
        # Predict location (mu) and scale (b) using the modified encoder
        loc, scale, _, prob_mono = self.encoder(batch["ref_img"], batch["ref_mask"])  # loc: (N, fW), scale: (N, fW)

        # Compute Laplace NLL loss using the predicted location and scale
        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        self.log("nll_loss-valid", nll_loss, prog_bar=True, logger=True, on_epoch=True)

        loss = nll_loss

        # Optionally add shape loss based on cosine similarity
        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            self.log("shape_loss-valid", shape_loss, prog_bar=True, logger=True, on_epoch=True)

            loss += shape_loss

        self.log("loss-valid", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss









class depth_net_metric3d_pl_old(pl.LightningModule):
    """
    lightning wrapper for the depth_net
    """# 
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
        self.encoder = depth_net_metric3d(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight# 
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer# 
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
        self.log("l1_loss-train", l1_loss, prog_bar=True, logger=True)# 
        loss = l1_loss# 
        if self.shape_loss_weight is not None:
            shape_loss = self.shape_loss_weight * (
                1 - F.cosine_similarity(d_mono, batch["ref_depth"], dim=-1).mean()
            )
            self.log("shape_loss-train", shape_loss, prog_bar=True, logger=True)# 
            loss += shape_loss# 
        self.log("loss-train", loss, prog_bar=True, logger=True)
        return loss# # # 
    def training_step(self, batch, batch_idx):
        d_mono, _, prob_mono = self.encoder(
            batch["ref_img"], batch["ref_mask"]
        )  # d_mono: (N, fWm), prob_mono: (N, fWm, D)# 
        # Check for NaNs in d_mono and batch["ref_depth"]
        if torch.isnan(d_mono).any() or torch.isnan(batch["ref_depth"]).any():
            print(f"NaNs found in d_mono or batch['ref_depth'] at batch_idx {batch_idx}")
            print("d_mono.shape: ", d_mono.shape)
            print("batch[ref_depth].shape : ", batch["ref_depth"].shape)
            return None# 
        #print(f"d_mono shape: {d_mono.shape}")  # Debug statement
        #print("batch[ref_depth] shape: ", batch["ref_depth"].shape)  # Debug statement# 
        l1_loss = F.l1_loss(d_mono, batch["ref_depth"])
        self.log("l1_loss-train", l1_loss, prog_bar=True, logger=True)# 
        loss = l1_loss# 
        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(d_mono, batch["ref_depth"], dim=-1).mean()
            
            # Check for NaNs in cosine_sim
            if torch.isnan(cosine_sim).any():
                print(f"NaNs found in cosine_similarity at batch_idx {batch_idx}")
                return None# 
            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            self.log("shape_loss-train", shape_loss, prog_bar=True, logger=True)# 
            loss += shape_loss# 
        self.log("loss-train", loss, prog_bar=True, logger=True)
        return loss# # # # 
    def validation_step(self, batch, batch_idx):
        d_mono, _, prob_mono = self.encoder(
            batch["ref_img"], batch["ref_mask"]
        )  # d_mono: (N, fWm), prob_mono: (N, fWm, D)
        #prob_mono = prob_mono.unsqueeze(1)  # (N, 1, fWm, D)
        #d_mono = d_mono.unsqueeze(1)  # (N, 1, fWm)# 
        #print(f"d_mono shape: {d_mono.shape}")  # Debug statement
        #print("batch[ref_depth] shape: ", batch["ref_depth"].shape)  # Debug statement# 
        l1_loss = F.l1_loss(d_mono, batch["ref_depth"])
        self.log("l1_loss-valid", l1_loss, prog_bar=True, logger=True, on_epoch=True)# 
        loss = l1_loss# 
        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(d_mono, batch["ref_depth"], dim=-1).mean()
            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            self.log("shape_loss-valid", shape_loss, prog_bar=True, logger=True, on_epoch=True)# 
            loss += shape_loss# 
        self.log("loss-valid", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss





class depth_net_depthanything_pl_original(pl.LightningModule):
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
        self.encoder = depth_net_depthanything(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

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
        )
        # d_mono: (N, fWm), prob_mono: (N, fWm, D)
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


class depth_net_depthanything_pl(pl.LightningModule):
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
        self.encoder = depth_net_depthanything(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight
        self.save_hyperparameters()

        # Placeholder to store metrics during the epoch
        self.training_metrics = []
        self.validation_metrics = []

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

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
        metrics = {"l1_loss": l1_loss}

        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(d_mono, batch["ref_depth"], dim=-1).mean()
            
            # Check for NaNs in cosine_sim
            if torch.isnan(cosine_sim).any():
                print(f"NaNs found in cosine_similarity at batch_idx {batch_idx}")
                return None

            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            self.log("shape_loss-train", shape_loss, prog_bar=True, logger=True)

            loss += shape_loss
            metrics["shape_loss"] = shape_loss

        metrics["loss"] = loss
        self.training_metrics.append(metrics)
        self.log_dict({f"{k}-train": v for k, v in metrics.items()}, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        # Compute averages for metrics
        avg_metrics = {key: torch.stack([m[key] for m in self.training_metrics]).mean() for key in self.training_metrics[0]}
        self.log_dict({f"avg_{key}-train": value for key, value in avg_metrics.items()}, prog_bar=True, logger=True)

        # Clear metrics storage
        self.training_metrics = []

    def validation_step(self, batch, batch_idx):
        d_mono, _, prob_mono = self.encoder(
            batch["ref_img"], batch["ref_mask"]
        )
        # d_mono: (N, fWm), prob_mono: (N, fWm, D)
        #prob_mono = prob_mono.unsqueeze(1)  # (N, 1, fWm, D)
        #d_mono = d_mono.unsqueeze(1)  # (N, 1, fWm)

        l1_loss = F.l1_loss(d_mono, batch["ref_depth"])
        self.log("l1_loss-valid", l1_loss, prog_bar=True, logger=True, on_epoch=True)
        loss = l1_loss
        metrics = {"l1_loss": l1_loss}

        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(d_mono, batch["ref_depth"], dim=-1).mean()
            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            self.log("shape_loss-valid", shape_loss, prog_bar=True, logger=True, on_epoch=True)
            loss += shape_loss
            metrics["shape_loss"] = shape_loss

        metrics["loss"] = loss
        self.validation_metrics.append(metrics)
        self.log_dict({f"{k}-valid": v for k, v in metrics.items()}, prog_bar=True, logger=True)        
        return loss
    
    def on_validation_epoch_end(self):
        # Compute averages for metrics
        avg_metrics = {key: torch.stack([m[key] for m in self.validation_metrics]).mean() for key in self.validation_metrics[0]}
        self.log_dict({f"avg_{key}-valid": value for key, value in avg_metrics.items()}, prog_bar=True, logger=True)

        # Clear metrics storage
        self.validation_metrics = []


        

class depth_net_depthanything_uncertainty_pl_original(pl.LightningModule):
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
        self.encoder = depth_net_depthanything_uncertainty(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def laplace_nll_loss(self, y_true, loc, scale):
        """Compute the negative log-likelihood loss for Laplace distribution."""
        return torch.mean(torch.abs(y_true - loc) / scale + torch.log(2 * scale))
    
    def gaussian_nll_loss(self, y_true, loc, scale): #gaussian_nll_loss
        """Compute the negative log-likelihood loss for a Gaussian distribution."""
        # Ensure scale is positive to avoid invalid operations
        eps = 1e-6  # small constant to ensure numerical stability
        scale = torch.clamp(scale, min=eps)
        # Compute the Gaussian NLL loss
        nll = 0.5 * ((y_true - loc) ** 2 / (scale**2) + torch.log(2 * torch.pi * scale**2))
        return torch.mean(nll)
    

    def training_step(self, batch, batch_idx):
        # Predict location (mu) and scale (b) using the modified encoder
        loc, scale, _, prob_mono = self.encoder(batch["ref_img"], batch["ref_mask"])  # loc: (N, fW), scale: (N, fW)

        # Check for NaNs in loc, scale, and batch["ref_depth"]
        if torch.isnan(loc).any() or torch.isnan(scale).any() or torch.isnan(batch["ref_depth"]).any():
            print(f"NaNs found in loc, scale, or batch['ref_depth'] at batch_idx {batch_idx}")
            return None

        # Compute Laplace NLL loss using the predicted location and scale
        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        self.log("nll_loss-train", nll_loss, prog_bar=True, logger=True)

        loss = nll_loss

        # Optionally add shape loss based on cosine similarity
        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            
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
        # Predict location (mu) and scale (b) using the modified encoder
        loc, scale, _, prob_mono = self.encoder(batch["ref_img"], batch["ref_mask"])  # loc: (N, fW), scale: (N, fW)

        # Compute Laplace NLL loss using the predicted location and scale
        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        self.log("nll_loss-valid", nll_loss, prog_bar=True, logger=True, on_epoch=True)

        loss = nll_loss

        # Optionally add shape loss based on cosine similarity
        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            self.log("shape_loss-valid", shape_loss, prog_bar=True, logger=True, on_epoch=True)

            loss += shape_loss

        self.log("loss-valid", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss
        

        
class depth_net_depthanything_uncertainty_pl(pl.LightningModule):
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
        self.encoder = depth_net_depthanything_uncertainty(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight
        self.save_hyperparameters()

        # Placeholder to store metrics during the epoch
        self.training_metrics = []
        self.validation_metrics = []

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def laplace_nll_loss(self, y_true, loc, scale):
        """Compute the negative log-likelihood loss for Laplace distribution."""
        return torch.mean(torch.abs(y_true - loc) / scale + torch.log(2 * scale))

    def gaussian_nll_loss(self, y_true, loc, scale):
        """Compute the negative log-likelihood loss for a Gaussian distribution."""
        eps = 1e-6
        scale = torch.clamp(scale, min=eps)
        nll = 0.5 * ((y_true - loc) ** 2 / (scale**2) + torch.log(2 * torch.pi * scale**2))
        return torch.mean(nll)

    def training_step(self, batch, batch_idx):
        loc, scale, _, prob_mono = self.encoder(batch["ref_img"], batch["ref_mask"])

        if torch.isnan(loc).any() or torch.isnan(scale).any() or torch.isnan(batch["ref_depth"]).any():
            print(f"NaNs found in loc, scale, or batch['ref_depth'] at batch_idx {batch_idx}")
            return None

        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        loss = nll_loss

        metrics = {"nll_loss": nll_loss}

        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            if torch.isnan(cosine_sim).any():
                print(f"NaNs found in cosine_similarity at batch_idx {batch_idx}")
                return None

            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            loss += shape_loss
            metrics["shape_loss"] = shape_loss

        metrics["loss"] = loss
        self.training_metrics.append(metrics)
        self.log_dict({f"{k}-train": v for k, v in metrics.items()}, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        # Compute averages for metrics
        avg_metrics = {key: torch.stack([m[key] for m in self.training_metrics]).mean() for key in self.training_metrics[0]}
        self.log_dict({f"avg_{key}-train": value for key, value in avg_metrics.items()}, prog_bar=True, logger=True)

        # Clear metrics storage
        self.training_metrics = []

    def validation_step(self, batch, batch_idx):
        loc, scale, _, prob_mono = self.encoder(batch["ref_img"], batch["ref_mask"])
        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        loss = nll_loss

        metrics = {"nll_loss": nll_loss}

        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            loss += shape_loss
            metrics["shape_loss"] = shape_loss

        metrics["loss"] = loss
        self.validation_metrics.append(metrics)
        self.log_dict({f"{k}-valid": v for k, v in metrics.items()}, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        # Compute averages for metrics
        avg_metrics = {key: torch.stack([m[key] for m in self.validation_metrics]).mean() for key in self.validation_metrics[0]}
        self.log_dict({f"avg_{key}-valid": value for key, value in avg_metrics.items()}, prog_bar=True, logger=True)

        # Clear metrics storage
        self.validation_metrics = []





class depth_net_depthanything_uncertainty_sem_pl(pl.LightningModule):
    """
    lightning wrapper for the depth_net
    """

    def __init__(
        self,
        shape_loss_weight=None,
        lambda_s = 1,
        temperature = 1,
        ce_weights = None, 
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
        self.encoder = depth_net_depthanything_uncertainty_sem_v2(
            d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, D=self.D
        )
        self.shape_loss_weight = shape_loss_weight
        self.lambda_s = lambda_s
        self.ce_weights = ce_weights
        self.temperature = temperature

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def combined_loss(self, y_true, loc, scale): 
        """Compute the negative log-likelihood loss for Laplace distribution."""
        return torch.mean(torch.abs(y_true - loc) / scale + torch.log(2 * scale))

#    def cross_entropy_loss(self, s_true, s_logits):
#        print(f"s_true.shape: {s_true.shape}")
#        print(f"s_logits.shape: {s_logits.shape}")
#
#        return F.cross_entropy(s_logits/self.temperature, s_true, weight = self.ce_weights)



#    def cross_entropy_loss(self, s_true, s_logits):
#        #print(f"s_true.shape: {s_true.shape}")  # [4, 90]
#        #print(f"s_logits.shape: {s_logits.shape}")  # [4, 90, 3]
#
#        # Ensure s_logits is float32
#        s_logits = s_logits.float()
#
#        # Ensure class weights are float32
#        if self.ce_weights is not None:
#            self.ce_weights = self.ce_weights.float()
#
#        # Flatten s_logits and s_true
#        s_logits = s_logits.view(-1, s_logits.size(-1))  # Reshape to [4 * 90, 3]
#        s_true = s_true.view(-1)  # Reshape to [4 * 90]
#
#        #print(f"Reshaped s_true.shape: {s_true.shape}")  # [4 * 90]
#        #print(f"Reshaped s_logits.shape: {s_logits.shape}")  # [4 * 90, 3]
#
#        # Apply cross entropy
#        return F.cross_entropy(s_logits / self.temperature, s_true, weight=self.ce_weights)



#    def cross_entropy_loss(self, s_true, s_logits):
#        #print(f"s_true.shape: {s_true.shape}")  # [4, 90]
#        #print(f"s_logits.shape: {s_logits.shape}")  # [4, 90, 3]
#        #print("Unique values in s_true:", torch.unique(s_true))  # Check unique values in s_true
#
#        # Ensure s_logits is float32
#        s_logits = s_logits.float()
#
#        # Ensure class weights are float32
#        if self.ce_weights is not None:
#            self.ce_weights = self.ce_weights.float()
#
#        # Flatten s_logits and s_true
#        s_logits = s_logits.view(-1, s_logits.size(-1))  # Reshape to [4 * 90, 3]
#        s_true = s_true.view(-1)  # Reshape to [4 * 90]
#
#        #print(f"Reshaped s_true.shape: {s_true.shape}")  # [4 * 90]
#        #print(f"Reshaped s_logits.shape: {s_logits.shape}")  # [4 * 90, 3]
#
#        # Apply cross entropy
#        return F.cross_entropy(s_logits / self.temperature, s_true, weight=self.ce_weights)



    def cross_entropy_loss(self, s_true, s_logits):
        # Ensure s_logits is float32 and within a stable range
        s_logits = s_logits.float()

        # Ensure class weights are float32
        if self.ce_weights is not None:
            self.ce_weights = self.ce_weights.float()
            assert torch.isfinite(self.ce_weights).all(), "Invalid class weights"

        # Reshape s_true to remove the last dimension (shape (4, 90, 1) -> (4, 90))
        s_true = s_true.squeeze(-1)

        # Ensure s_true is in the valid range of class indices
        assert s_true.min() >= 0 and s_true.max() < s_logits.size(-1), "Invalid target labels"

        # Transpose s_logits from (batch_size, sequence_length, num_classes) to (batch_size, num_classes, sequence_length)
        s_logits = s_logits.permute(0, 2, 1)

        # Compute the cross-entropy loss across the batch and sequence dimensions
        loss = F.cross_entropy(s_logits, s_true, weight=self.ce_weights)
        
        return loss



    def laplace_nll_loss(self, y_true, loc, scale):
        """Compute the negative log-likelihood loss for Laplace distribution."""
        return torch.mean(torch.abs(y_true - loc) / scale + torch.log(2 * scale))


    def training_step(self, batch, batch_idx):
        # Predict location (mu) and scale (b) using the modified encoder
        loc, scale, _, prob_mono, class_logits, class_probs = self.encoder(batch["ref_img"], batch["ref_mask"])  # loc: (N, fW), scale: (N, fW)

        # Check for NaNs in loc, scale, and batch["ref_depth"]
        if torch.isnan(loc).any() or torch.isnan(scale).any() or torch.isnan(batch["ref_depth"]).any():
            print(f"NaNs found in loc, scale, or batch['ref_depth'] at batch_idx {batch_idx}")
            return None

        # Compute combined loss
        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        self.log("nll_loss-train", nll_loss, prog_bar=True, logger=True)
        ce_loss = self.lambda_s * self.cross_entropy_loss(batch["ref_semantic"], class_logits)
        self.log("ce_loss-train", ce_loss, prog_bar=True, logger=True)
        combined_loss = nll_loss + ce_loss
        self.log("combined_loss-train", combined_loss, prog_bar=True, logger=True)

        loss = combined_loss

        # Optionally add shape loss based on cosine similarity
        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            
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
        # Predict location (mu) and scale (b) using the modified encoder
        loc, scale, _, prob_mono, class_logits, class_probs = self.encoder(batch["ref_img"], batch["ref_mask"])  # loc: (N, fW), scale: (N, fW)

        # Compute combined loss
        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        self.log("nll_loss-valid", nll_loss, prog_bar=True, logger=True)
        ce_loss = self.lambda_s * self.cross_entropy_loss(batch["ref_semantic"], class_logits)
        self.log("ce_loss-valid", ce_loss, prog_bar=True, logger=True)
        combined_loss = nll_loss + ce_loss
        self.log("combined_loss-valid", combined_loss, prog_bar=True, logger=True)

        loss = combined_loss

        # Optionally add shape loss based on cosine similarity
        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            self.log("shape_loss-valid", shape_loss, prog_bar=True, logger=True, on_epoch=True)

            loss += shape_loss

        self.log("loss-valid", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss





class semantic_net_depthanything_pl(pl.LightningModule):
    """
    lightning wrapper for the depth_net
    """

    def __init__(
        self,
        ce_weights = None, 
        lr=1e-3,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.encoder = semantic_net_depthanything()
        self.ce_weights = ce_weights

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

#    def cross_entropy_loss(self, s_true, s_logits):
#
#        # Ensure s_logits is float32
#        s_logits = s_logits.float()
#        s_logits = torch.clamp(s_logits, min=-1e9, max=1e9)
#
#        # Ensure class weights are float32
#        if self.ce_weights is not None:
#            self.ce_weights = self.ce_weights.float()
#
#        # Flatten s_logits and s_true
#        s_logits = s_logits.view(-1, s_logits.size(-1))  # Reshape to [4 * 90, 3]
#        s_true = s_true.view(-1)  # Reshape to [4 * 90]
#
#        # Apply cross entropy
#        return F.cross_entropy(s_logits, s_true, weight=self.ce_weights)
#
#    def training_step(self, batch, batch_idx):
#        # Predict location (mu) and scale (b) using the modified encoder
#        class_logits, class_probs = self.encoder(batch["ref_img"], batch["ref_mask"])  # loc: (N, fW), scale: (N, fW)
#
#        # Compute combined loss
#        ce_loss = self.cross_entropy_loss(batch["ref_semantic"], class_logits)
#        self.log("ce_loss-train", ce_loss, prog_bar=True, logger=True)
#        loss = ce_loss
#
#        self.log("loss-train", loss, prog_bar=True, logger=True)
#        return loss
# 
#    def validation_step(self, batch, batch_idx):
#        # Predict location (mu) and scale (b) using the modified encoder
#        class_logits, class_probs = self.encoder(batch["ref_img"], batch["ref_mask"])  # loc: (N, fW), scale: (N, fW)
#
#        # Compute combined loss
#        ce_loss = self.cross_entropy_loss(batch["ref_semantic"], class_logits)
#        self.log("ce_loss-valid", ce_loss, prog_bar=True, logger=True)
#        loss = ce_loss
#
#        self.log("loss-valid", loss, prog_bar=True, logger=True, on_epoch=True)
#        return loss

#    def cross_entropy_loss(self, s_true, s_logits):
#        # Ensure s_logits is float32 and within a stable range
#        s_logits = s_logits.float()
#        s_logits = torch.clamp(s_logits, min=-1e9, max=1e9)
#
#        # Ensure class weights are float32
#        if self.ce_weights is not None:
#            self.ce_weights = self.ce_weights.float()
#            assert torch.isfinite(self.ce_weights).all(), "Invalid class weights"
#
#        # Flatten s_logits and s_true
#        s_logits = s_logits.view(-1, s_logits.size(-1))  # Reshape to [4 * 90, 3]
#        s_true = s_true.view(-1)  # Reshape to [4 * 90]
#
#        # Check target labels are in valid range
#        assert s_true.min() >= 0 and s_true.max() < s_logits.size(-1), "Invalid target labels"
#
#        # Apply cross entropy
#        return F.cross_entropy(s_logits, s_true, weight=self.ce_weights)


#    def cross_entropy_loss(self, s_true, s_logits):
#        # Ensure s_logits is float32 and within a stable range
#        s_logits = s_logits.float()
#
#        # Ensure class weights are float32
#        if self.ce_weights is not None:
#            self.ce_weights = self.ce_weights.float()
#            assert torch.isfinite(self.ce_weights).all(), "Invalid class weights"
#
#        # Flatten s_logits and s_true
#        s_logits = s_logits.view(-1, s_logits.size(-1))  # Reshape to [4 * 90, 3]
#        s_true = s_true.view(-1)  # Reshape to [4 * 90]
#
#        # Check target labels are in valid range
#        assert s_true.min() >= 0 and s_true.max() < s_logits.size(-1), "Invalid target labels"
#
#        # Apply cross entropy
#        return F.cross_entropy(s_logits, s_true, weight=self.ce_weights)


    def cross_entropy_loss(self, s_true, s_logits):
        # Ensure s_logits is float32 and within a stable range
        s_logits = s_logits.float()

        # Ensure class weights are float32
        if self.ce_weights is not None:
            self.ce_weights = self.ce_weights.float()
            assert torch.isfinite(self.ce_weights).all(), "Invalid class weights"

        # Reshape s_true to remove the last dimension (shape (4, 90, 1) -> (4, 90))
        s_true = s_true.squeeze(-1)

        # Ensure s_true is in the valid range of class indices
        assert s_true.min() >= 0 and s_true.max() < s_logits.size(-1), "Invalid target labels"

        # Transpose s_logits from (batch_size, sequence_length, num_classes) to (batch_size, num_classes, sequence_length)
        s_logits = s_logits.permute(0, 2, 1)

        # Compute the cross-entropy loss across the batch and sequence dimensions
        loss = F.cross_entropy(s_logits, s_true, weight=self.ce_weights)
        
        return loss


    def training_step(self, batch, batch_idx):
        # Check input images and masks for NaNs or Infs
        assert torch.isfinite(batch["ref_img"]).all(), "Invalid values in input images"
        assert torch.isfinite(batch["ref_mask"]).all(), "Invalid values in input masks"
    
    
        # Predict location (mu) and scale (b) using the modified encoder
        class_logits, class_probs = self.encoder(batch["ref_img"], batch["ref_mask"])

        # Ensure class_logits are finite
        assert torch.isfinite(class_logits).all(), "Invalid logits"

        # Compute and log the loss
        ce_loss = self.cross_entropy_loss(batch["ref_semantic"], class_logits)
        self.log("ce_loss-train", ce_loss, prog_bar=True, logger=True)
        loss = ce_loss

        # Log and return the loss
        self.log("loss-train", loss, prog_bar=True, logger=True)
        return loss


#    def validation_step(self, batch, batch_idx):
#        # Predict location (mu) and scale (b) using the modified encoder
#        class_logits, class_probs = self.encoder(batch["ref_img"], batch["ref_mask"])
#
#        # Ensure class_logits are finite
#        assert torch.isfinite(class_logits).all(), "Invalid logits in validation step"
#
#        # Compute and log the loss
#        ce_loss = self.cross_entropy_loss(batch["ref_semantic"], class_logits)
#        self.log("ce_loss-valid", ce_loss, prog_bar=True, logger=True)
#        loss = ce_loss
#
#        # Log and return the loss
#        self.log("loss-valid", loss, prog_bar=True, logger=True, on_epoch=True)
#        return loss


    def validation_step(self, batch, batch_idx):
        # Validate input
        assert torch.isfinite(batch["ref_img"]).all(), "Invalid values in input images"
        assert torch.isfinite(batch["ref_mask"]).all(), "Invalid values in input masks"
        
        # Forward pass
        class_logits, class_probs = self.encoder(batch["ref_img"], batch["ref_mask"])
        
        # Ensure logits are finite
        assert torch.isfinite(class_logits).all(), "Invalid logits in validation step"
        
        # Compute loss
        ce_loss = self.cross_entropy_loss(batch["ref_semantic"], class_logits)
        loss = ce_loss

        # Logging
        self.log("ce_loss-valid", ce_loss, prog_bar=True, logger=True)
        self.log("loss-valid", loss, prog_bar=True, logger=True, on_epoch=True)
        
        return loss
