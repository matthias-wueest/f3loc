from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import *


import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

def localize(
    desdf: torch.tensor, rays: torch.tensor, orn_slice=36, return_np=True, lambd=40
) -> Tuple[torch.tensor]:
    """
    Localize in the desdf according to the rays
    Input:
        desdf: (H, W, O), counter clockwise
        rays: (V,) from left to right (clockwise)
        orn_slice: number of orientations
        return_np: return as ndarray instead of torch.tensor
        lambd: parameter for likelihood
    Output:
        prob_vol: probability volume (H, W, O), ndarray
        prob_dist: probability distribution, (H, W) maxpool the prob_vol along orientation, ndarray
        orientations: orientation with max likelihood at each position, (H, W), ndarray
        pred: (3, ) predicted state [x,y,theta], ndarray
    """

    # flip the ray, to make rotation direction mathematically positive
    rays = torch.flip(rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    # expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))

    # circular pad the desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    # probability is -l1norm
    prob_vol = torch.stack(
        [
            -torch.norm(pad_desdf[:, :, i : i + V] - rays, p=1.0, dim=2)
            for i in range(O)
        ],
        dim=2,
    )  # (H,W,O)
    prob_vol = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive
    
    ###
    total_sum = torch.sum(prob_vol)
    prob_vol = prob_vol / total_sum
    ###

    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # get the prediction
    prob_dist_cpu = prob_dist.cpu().numpy()  # Move the tensor to the CPU and convert to numpy
    pred_y, pred_x = np.unravel_index(np.argmax(prob_dist_cpu), prob_dist_cpu.shape)
    device = prob_dist.device  # Use the same device as prob_dist
    pred_x = torch.tensor(pred_x, device=device)
    pred_y = torch.tensor(pred_y, device=device)

    orn = orientations[pred_y, pred_x]
    # from orientation indices to radians
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x.unsqueeze(0), pred_y.unsqueeze(0), orn.unsqueeze(0)))

    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )

def localize_noflip(
    desdf: torch.tensor, rays: torch.tensor, orn_slice=36, return_np=True, lambd=40
) -> Tuple[torch.tensor]:
    """
    Localize in the desdf according to the rays
    Input:
        desdf: (H, W, O), counter clockwise
        rays: (V,) from left to right (clockwise)
        orn_slice: number of orientations
        return_np: return as ndarray instead of torch.tensor
        lambd: parameter for likelihood
    Output:
        prob_vol: probability volume (H, W, O), ndarray
        prob_dist: probability distribution, (H, W) maxpool the prob_vol along orientation, ndarray
        orientations: orientation with max likelihood at each position, (H, W), ndarray
        pred: (3, ) predicted state [x,y,theta], ndarray
    """

    # flip the ray, to make rotation direction mathematically positive
    #rays = torch.flip(rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    # expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))

    # circular pad the desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

#    # probability is -l1norm
#    prob_vol = torch.stack(
#        [
#            -torch.norm(pad_desdf[:, :, i : i + V] - rays, p=1.0, dim=2)
#            for i in range(O)
#        ],
#        dim=2,
#    )  # (H,W,O)

    prob_vol = torch.stack(
        [
            -torch.sum(
                torch.clamp(
                    torch.abs(pad_desdf[:, :, i : i + V] - rays), max=5
                ),
                dim=2
            ) / V * 11
            for i in range(O)
        ],
        dim=2,
    )

    del pad_desdf  # Free memory for this variable if no longer used
    prob_vol = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive
    ###
    total_sum = torch.sum(prob_vol)
    prob_vol = prob_vol / total_sum
    ###


    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # get the prediction
    prob_dist_cpu = prob_dist.cpu().numpy()  # Move the tensor to the CPU and convert to numpy
    pred_y, pred_x = np.unravel_index(np.argmax(prob_dist_cpu), prob_dist_cpu.shape)
    #pred_y, pred_x = torch.where(prob_dist_cpu == prob_dist_cpu.max())
    device = prob_dist.device  # Use the same device as prob_dist
    pred_x = torch.tensor(pred_x, device=device)
    pred_y = torch.tensor(pred_y, device=device)

    orn = orientations[pred_y, pred_x]
    # from orientation indices to radians
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x.unsqueeze(0), pred_y.unsqueeze(0), orn.unsqueeze(0)))

    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )


def localize_noflip_uncertainty(
    desdf: torch.tensor, rays: torch.tensor, scales_rays: torch.tensor, orn_slice=36, return_np=True, lambd=40
) -> Tuple[torch.tensor]:
    """
    Localize in the desdf according to the rays
    Input:
        desdf: (H, W, O), counter clockwise
        rays: (V,) from left to right (clockwise)
        orn_slice: number of orientations
        return_np: return as ndarray instead of torch.tensor
        lambd: parameter for likelihood
    Output:
        prob_vol: probability volume (H, W, O), ndarray
        prob_dist: probability distribution, (H, W) maxpool the prob_vol along orientation, ndarray
        orientations: orientation with max likelihood at each position, (H, W), ndarray
        pred: (3, ) predicted state [x,y,theta], ndarray
    """

    # flip the ray, to make rotation direction mathematically positive
    #rays = torch.flip(rays, [0])
    #scales_rays = torch.flip(scales_rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    # expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))
    scales_rays = scales_rays.reshape((1, 1, -1))

    # circular pad the desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

#    # probability is -l1norm
#    prob_vol = torch.stack(
#        [
#            -torch.norm(pad_desdf[:, :, i : i + V] - rays, p=1.0, dim=2)
#            for i in range(O)
#        ],
#        dim=2,
#    )  # (H,W,O)

# ### Laplacian
#     prob_vol = torch.stack(
#         [
#             -torch.sum(
#                 torch.clamp(
#                     torch.abs((pad_desdf[:, :, i : i + V] - rays)/scales_rays), max=5 #2.5
#                 ),
#                 dim=2
#             ) / V * 11
#             for i in range(O)
#         ],
#         dim=2,
#     )

    # Gaussian
    prob_vol = torch.stack(
        [
            -torch.sum(
                torch.clamp(
                    (pad_desdf[:, :, i : i + V] - rays)**2/(2*scales_rays**2), max=5 #2.5
                ),
                dim=2
            ) / V * 11
            for i in range(O)
        ],
        dim=2,
    )

    
    del pad_desdf  # Free memory for this variable if no longer used
    prob_vol = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive

    ###
    total_sum = torch.sum(prob_vol)
    prob_vol = prob_vol / total_sum
    ###


    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # get the prediction
    prob_dist_cpu = prob_dist.cpu().numpy()  # Move the tensor to the CPU and convert to numpy
    pred_y, pred_x = np.unravel_index(np.argmax(prob_dist_cpu), prob_dist_cpu.shape)
    #pred_y, pred_x = torch.where(prob_dist_cpu == prob_dist_cpu.max())
    device = prob_dist.device  # Use the same device as prob_dist
    pred_x = torch.tensor(pred_x, device=device)
    pred_y = torch.tensor(pred_y, device=device)

    orn = orientations[pred_y, pred_x]
    # from orientation indices to radians
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x.unsqueeze(0), pred_y.unsqueeze(0), orn.unsqueeze(0)))

    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )



def localize_uncertainty(
    desdf: torch.tensor, rays: torch.tensor, scales_rays: torch.tensor, orn_slice=36, return_np=True, lambd=40
) -> Tuple[torch.tensor]:

    # flip the ray, to make rotation direction mathematically positive
    rays = torch.flip(rays, [0])
    scales_rays = torch.flip(scales_rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    dv = int(360/orn_slice)
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi
    cos_angles = torch.cos(torch.tensor(angles)).to(rays.device)
    cos_angles = torch.flip(cos_angles, [0])


    # expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))
    scales_rays = scales_rays.reshape((1, 1, -1))

    # circular pad the desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    #i = 1
    #print(torch.norm((pad_desdf[:, :, i : i + V]*cos_angles - rays)/scales_rays, p=1.0, dim=2).shape)
    #print(torch.log(2*scales_rays).shape)
    
    ### Depth probability
    # probability is -l1norm
#    prob_vol_depth = torch.stack(
#        [
#            -torch.norm(((pad_desdf[:, :, i : i + V] - rays)*cos_angles)/scales_rays, p=1.0, dim=2) - torch.sum(torch.log(2*scales_rays))
#            for i in range(O)
#        ],
#        dim=2,
#    )  # (H,W,O)

    prob_vol_depth = torch.stack(
        [
            -torch.norm(((pad_desdf[:, :, i : i + V] - rays))/scales_rays, p=1.0, dim=2)
            for i in range(O)
        ],
        dim=2,
    )  # (H,W,O)

    prob_vol_depth = torch.exp(prob_vol_depth / lambd)  # NOTE: here make prob positive

    ###
    prob_vol = prob_vol_depth
    total_sum = torch.sum(prob_vol)
    prob_vol = prob_vol / total_sum
    ###

    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # get the prediction
    prob_dist_cpu = prob_dist.cpu().numpy()  # Move the tensor to the CPU and convert to numpy
    pred_y, pred_x = np.unravel_index(np.argmax(prob_dist_cpu), prob_dist_cpu.shape)
    device = prob_dist.device  # Use the same device as prob_dist
    pred_x = torch.tensor(pred_x, device=device)
    pred_y = torch.tensor(pred_y, device=device)

    orn = orientations[pred_y, pred_x]
    # from orientation indices to radians
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x.unsqueeze(0), pred_y.unsqueeze(0), orn.unsqueeze(0)))

    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )



#def localize_uncertainty(
#    desdf: torch.tensor, desdf_semantics: torch.tensor, rays: torch.tensor, scales_rays: torch.tensor, semantics_rays: torch.tensor, orn_slice=36, return_np=True, lambd=40
#) -> Tuple[torch.tensor]:
#
#    # flip the ray, to make rotation direction mathematically positive
#    rays = torch.flip(rays, [0])
#    scales_rays = torch.flip(scales_rays, [0])
#    O = desdf.shape[2]
#    V = rays.shape[0]
#    dv = int(360/orn_slice)
#    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi
#    cos_angles = torch.cos(torch.tensor(angles)).to(rays.device)
#    cos_angles = torch.flip(cos_angles, [0])
#
#
#    # expand rays to have the same dimension as desdf
#    rays = rays.reshape((1, 1, -1))
#    scales_rays = scales_rays.reshape((1, 1, -1))
#    semantics_rays = semantics_rays.reshape((1, 1, -1))
#
#    # circular pad the desdf
#    pad_front = V // 2
#    pad_back = V - pad_front
#    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")
#    desdf_semantics = F.pad(desdf_semantics, [pad_front, pad_back], mode="circular")
#
#
#    #i = 1
#    #print(torch.norm((pad_desdf[:, :, i : i + V]*cos_angles - rays)/scales_rays, p=1.0, dim=2).shape)
#    #print(torch.log(2*scales_rays).shape)
#    
#    ### Depth probability
#    # probability is -l1norm
#    prob_vol_depth = torch.stack(
#        [
#            -torch.norm(((pad_desdf[:, :, i : i + V] - rays)*cos_angles)/scales_rays, p=1.0, dim=2) - torch.sum(torch.log(2*scales_rays))
#            for i in range(O)
#        ],
#        dim=2,
#    )  # (H,W,O)
#
#    prob_vol_depth = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive
#    
#
#    ### Semantics probability
#    prob_vol_semantic = torch.stack(
#        [
#            #-torch.norm(((pad_desdf[:, :, i : i + V] - rays)*cos_angles)/scales_rays, p=1.0, dim=2) - torch.sum(torch.log(2*scales_rays))
#            torch.prod(semantics_rays[pad_desdf[:, :, i : i + V] - 1])
#            for i in range(O)
#        ],
#        dim=2,
#    )  # (H,W,O)
#
#
#    ### Combined probability
#    prob_vol = prob_vol_depth * prob_vol_semantic
#    ###
#    total_sum = torch.sum(prob_vol)
#    prob_vol = prob_vol / total_sum
#    ###
#
#    # maxpooling
#    prob_dist, orientations = torch.max(prob_vol, dim=2)
#
#    # get the prediction
#    prob_dist_cpu = prob_dist.cpu().numpy()  # Move the tensor to the CPU and convert to numpy
#    pred_y, pred_x = np.unravel_index(np.argmax(prob_dist_cpu), prob_dist_cpu.shape)
#    device = prob_dist.device  # Use the same device as prob_dist
#    pred_x = torch.tensor(pred_x, device=device)
#    pred_y = torch.tensor(pred_y, device=device)
#
#    orn = orientations[pred_y, pred_x]
#    # from orientation indices to radians
#    orn = orn / orn_slice * 2 * torch.pi
#    pred = torch.cat((pred_x.unsqueeze(0), pred_y.unsqueeze(0), orn.unsqueeze(0)))
#
#    if return_np:
#        return (
#            prob_vol.detach().cpu().numpy(),
#            prob_dist.detach().cpu().numpy(),
#            orientations.detach().cpu().numpy(),
#            pred.detach().cpu().numpy(),
#        )
#    else:
#        return (
#            prob_vol.to(torch.float32).detach().cpu(),
#            prob_dist.to(torch.float32).detach().cpu(),
#            orientations.to(torch.float32).detach().cpu(),
#            pred.to(torch.float32).detach().cpu(),
#        )


def localize_noflip_uncertainty_semantics(
    desdf: torch.tensor, desdf_semantics: torch.tensor, rays: torch.tensor, scales_rays: torch.tensor, semantics_rays: torch.tensor, orn_slice=36, return_np=True, lambd=40
) -> Tuple[torch.tensor]:
    """
    Localize in the desdf according to the rays
    Input:
        desdf: (H, W, O), counter clockwise
        rays: (V,) from left to right (clockwise)
        orn_slice: number of orientations
        return_np: return as ndarray instead of torch.tensor
        lambd: parameter for likelihood
    Output:
        prob_vol: probability volume (H, W, O), ndarray
        prob_dist: probability distribution, (H, W) maxpool the prob_vol along orientation, ndarray
        orientations: orientation with max likelihood at each position, (H, W), ndarray
        pred: (3, ) predicted state [x,y,theta], ndarray
    """

    # flip the ray, to make rotation direction mathematically positive
    #rays = torch.flip(rays, [0])
    #scales_rays = torch.flip(scales_rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    H, W = desdf.shape[0], desdf.shape[1]  # spatial dimensions (H, W)
    # expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))
    scales_rays = scales_rays.reshape((1, 1, -1))

    # circular pad the desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    # probability is -l1norm
    prob_vol = torch.stack(
        [
            -torch.sum(
                torch.clamp(
                    torch.abs((pad_desdf[:, :, i : i + V] - rays)/scales_rays), max=5 #2.5
                ),
                dim=2
            ) / V * 11
            for i in range(O)
        ],
        dim=2,
    )
    
    del pad_desdf  # Free memory for this variable if no longer used
    prob_vol = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive

    ###
    total_sum = torch.sum(prob_vol)
    prob_vol = prob_vol / total_sum
    ###

    # #### Semantics
    # pad_desdf_semantics = F.pad(desdf_semantics - 1, [pad_front, pad_back], mode="circular")
    # prob_vol_semantics = torch.stack(
    #     [
    #         epsilon = 1e-8
    #         torch.sum(torch.log(semantics_rays[:, pad_desdf_semantics[:, :, i : i + V]]) + epsilon, dim=2)
    #         for i in range(O)
    #     ],
    #     dim=2,
    # )
    # del pad_desdf_semantics 
    # prob_vol_semantics = torch.exp(prob_vol_semantics)
    # #####

    #### Semantics Probability Volume ####

    # Circular pad desdf_semantics (contains the true class labels)
    pad_desdf_semantics = F.pad(desdf_semantics - 1, [pad_front, pad_back], mode="circular").to(torch.int64)

    # Epsilon to prevent log(0)
    epsilon = 1e-8

#    # For each orientation, gather the probabilities of the true semantic class from semantics_rays
#    prob_vol_semantics = torch.stack(
#        [
#            torch.sum(
#                torch.log(
#                    torch.gather(
#                        semantics_rays.unsqueeze(0).unsqueeze(0).expand(H, W, -1, -1),  # Expand semantics_rays to match (H, W, V, 3)
#                        3,  # Index along the class dimension (semantic classes)
#                        pad_desdf_semantics[:, :, i : i + V].unsqueeze(-1)  # Index true class (H, W, V, 1)
#                    ).squeeze(-1) + epsilon  # Remove the singleton dimension and add epsilon for stability (H, W, V)
#                ),
#                dim=2  # Sum log-probabilities over the ray dimension (V)
#            )
#            for i in range(O)
#        ],
#        dim=2
#    )    
#
#    # Apply exponential to convert log-probabilities back to probability space
#    prob_vol_semantics = torch.exp(prob_vol_semantics)


    # For each orientation, gather the probabilities of the true semantic class from semantics_rays
    prob_vol_semantics = torch.stack(
        [
            torch.sum(
                    torch.gather(
                        semantics_rays.unsqueeze(0).unsqueeze(0).expand(H, W, -1, -1),  # Expand semantics_rays to match (H, W, V, 3)
                        3,  # Index along the class dimension (semantic classes)
                        pad_desdf_semantics[:, :, i : i + V].unsqueeze(-1)  # Index true class (H, W, V, 1)
                    ).squeeze(-1) + epsilon,  # Remove the singleton dimension and add epsilon for stability (H, W, V)
                dim=2  # Sum log-probabilities over the ray dimension (V)
            )
            for i in range(O)
        ],
        dim=2
    )  


    # Apply exponential to convert log-probabilities back to probability space
    lambd_sem = 100
    prob_vol_semantics = torch.exp(prob_vol_semantics / lambd_sem)
    total_sum_semantics = torch.sum(prob_vol_semantics)
    prob_vol_semantics = prob_vol_semantics / total_sum_semantics
    ###


    # Combine both probability volumes
    prob_vol = prob_vol * prob_vol_semantics
    total_sum = torch.sum(prob_vol)
    prob_vol = prob_vol / total_sum
    ####


    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # get the prediction
    prob_dist_cpu = prob_dist.cpu().numpy()  # Move the tensor to the CPU and convert to numpy
    pred_y, pred_x = np.unravel_index(np.argmax(prob_dist_cpu), prob_dist_cpu.shape)
    #pred_y, pred_x = torch.where(prob_dist_cpu == prob_dist_cpu.max())
    device = prob_dist.device  # Use the same device as prob_dist
    pred_x = torch.tensor(pred_x, device=device)
    pred_y = torch.tensor(pred_y, device=device)

    orn = orientations[pred_y, pred_x]
    # from orientation indices to radians
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x.unsqueeze(0), pred_y.unsqueeze(0), orn.unsqueeze(0)))

    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )




def localize_noflip_semantics(
    desdf: torch.tensor, desdf_semantics: torch.tensor, rays: torch.tensor, semantics_rays: torch.tensor, orn_slice=36, return_np=True, lambd=40
) -> Tuple[torch.tensor]:
    """
    Localize in the desdf according to the rays
    Input:
        desdf: (H, W, O), counter clockwise
        rays: (V,) from left to right (clockwise)
        orn_slice: number of orientations
        return_np: return as ndarray instead of torch.tensor
        lambd: parameter for likelihood
    Output:
        prob_vol: probability volume (H, W, O), ndarray
        prob_dist: probability distribution, (H, W) maxpool the prob_vol along orientation, ndarray
        orientations: orientation with max likelihood at each position, (H, W), ndarray
        pred: (3, ) predicted state [x,y,theta], ndarray
    """

    # flip the ray, to make rotation direction mathematically positive
    #rays = torch.flip(rays, [0])
    #scales_rays = torch.flip(scales_rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    H, W = desdf.shape[0], desdf.shape[1]  # spatial dimensions (H, W)
    # expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))

    # circular pad the desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    # probability is -l1norm
    prob_vol = torch.stack(
        [
            -torch.sum(
                torch.clamp(
                    torch.abs(pad_desdf[:, :, i : i + V] - rays), max=5
                ),
                dim=2
            ) / V * 11
            for i in range(O)
        ],
        dim=2,
    )
    
    del pad_desdf  # Free memory for this variable if no longer used
    prob_vol = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive

    ###
    total_sum = torch.sum(prob_vol)
    prob_vol = prob_vol / total_sum
    ###

    # #### Semantics
    # pad_desdf_semantics = F.pad(desdf_semantics - 1, [pad_front, pad_back], mode="circular")
    # prob_vol_semantics = torch.stack(
    #     [
    #         epsilon = 1e-8
    #         torch.sum(torch.log(semantics_rays[:, pad_desdf_semantics[:, :, i : i + V]]) + epsilon, dim=2)
    #         for i in range(O)
    #     ],
    #     dim=2,
    # )
    # del pad_desdf_semantics 
    # prob_vol_semantics = torch.exp(prob_vol_semantics)
    # #####

    #### Semantics Probability Volume ####

    # Circular pad desdf_semantics (contains the true class labels)
    pad_desdf_semantics = F.pad(desdf_semantics - 1, [pad_front, pad_back], mode="circular").to(torch.int64)

    # Epsilon to prevent log(0)
    epsilon = 1e-8

#    # For each orientation, gather the probabilities of the true semantic class from semantics_rays
#    prob_vol_semantics = torch.stack(
#        [
#            torch.sum(
#                torch.log(
#                    torch.gather(
#                        semantics_rays.unsqueeze(0).unsqueeze(0).expand(H, W, -1, -1),  # Expand semantics_rays to match (H, W, V, 3)
#                        3,  # Index along the class dimension (semantic classes)
#                        pad_desdf_semantics[:, :, i : i + V].unsqueeze(-1)  # Index true class (H, W, V, 1)
#                    ).squeeze(-1) + epsilon  # Remove the singleton dimension and add epsilon for stability (H, W, V)
#                ),
#                dim=2  # Sum log-probabilities over the ray dimension (V)
#            )
#            for i in range(O)
#        ],
#        dim=2
#    )    


    # For each orientation, gather the probabilities of the true semantic class from semantics_rays
    prob_vol_semantics = torch.stack(
        [
            torch.sum(
                    torch.gather(
                        semantics_rays.unsqueeze(0).unsqueeze(0).expand(H, W, -1, -1),  # Expand semantics_rays to match (H, W, V, 3)
                        3,  # Index along the class dimension (semantic classes)
                        pad_desdf_semantics[:, :, i : i + V].unsqueeze(-1)  # Index true class (H, W, V, 1)
                    ).squeeze(-1) + epsilon,  # Remove the singleton dimension and add epsilon for stability (H, W, V)
                dim=2  # Sum log-probabilities over the ray dimension (V)
            )
            for i in range(O)
        ],
        dim=2
    )  


    # Apply exponential to convert log-probabilities back to probability space
    lambd_sem = 100
    prob_vol_semantics = torch.exp(prob_vol_semantics / lambd_sem)
    total_sum_semantics = torch.sum(prob_vol_semantics)
    prob_vol_semantics = prob_vol_semantics / total_sum_semantics
    ###


    # Combine both probability volumes
    prob_vol = prob_vol * prob_vol_semantics
    total_sum = torch.sum(prob_vol)
    prob_vol = prob_vol / total_sum
    ####


    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # get the prediction
    prob_dist_cpu = prob_dist.cpu().numpy()  # Move the tensor to the CPU and convert to numpy
    pred_y, pred_x = np.unravel_index(np.argmax(prob_dist_cpu), prob_dist_cpu.shape)
    #pred_y, pred_x = torch.where(prob_dist_cpu == prob_dist_cpu.max())
    device = prob_dist.device  # Use the same device as prob_dist
    pred_x = torch.tensor(pred_x, device=device)
    pred_y = torch.tensor(pred_y, device=device)

    orn = orientations[pred_y, pred_x]
    # from orientation indices to radians
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x.unsqueeze(0), pred_y.unsqueeze(0), orn.unsqueeze(0)))

    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )




def localize_uncertainty_bkp20240930(
    desdf: torch.tensor, rays: torch.tensor, scales_rays: torch.tensor, orn_slice=36, return_np=True, lambd=40
) -> Tuple[torch.tensor]:

    # flip the ray, to make rotation direction mathematically positive
    rays = torch.flip(rays, [0])
    scales_rays = torch.flip(scales_rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    # expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))
    scales_rays = scales_rays.reshape((1, 1, -1))

    # circular pad the desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    # probability is -l1norm
    prob_vol = torch.stack(
        [
            -torch.norm((pad_desdf[:, :, i : i + V] - rays)/scales_rays, p=1.0, dim=2)
            for i in range(O)
        ],
        dim=2,
    )  # (H,W,O)
    prob_vol = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive
    
    ###
    total_sum = torch.sum(prob_vol)
    prob_vol = prob_vol / total_sum
    ###

    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # get the prediction
    prob_dist_cpu = prob_dist.cpu().numpy()  # Move the tensor to the CPU and convert to numpy
    pred_y, pred_x = np.unravel_index(np.argmax(prob_dist_cpu), prob_dist_cpu.shape)
    device = prob_dist.device  # Use the same device as prob_dist
    pred_x = torch.tensor(pred_x, device=device)
    pred_y = torch.tensor(pred_y, device=device)

    orn = orientations[pred_y, pred_x]
    # from orientation indices to radians
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x.unsqueeze(0), pred_y.unsqueeze(0), orn.unsqueeze(0)))

    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )















# def localize(
#     desdf: torch.tensor, rays: torch.tensor, orn_slice=36, return_np=True, lambd=40
# ) -> Tuple[torch.tensor]:
#     """
#     Localize in the desdf according to the rays
#     Input:
#         desdf: (H, W, O), counter clockwise
#         rays: (V,) from left to right (clockwise)
#         orn_slice: number of orientations
#         return_np: return as ndarray instead of torch.tensor
#         lambd: parameter for likelihood
#     Output:
#         prob_vol: probability volume (H, W, O), ndarray
#         prob_dist: probability distribution, (H, W) maxpool the prob_vol along orientation, ndarray
#         orientations: orientation with max likelihood at each position, (H, W), ndarray
#         pred: (3, ) predicted state [x,y,theta], ndarray
#     """
# 
#     # flip the ray, to make rotation direction mathematically positive
#     rays = torch.flip(rays, [0])
#     O = desdf.shape[2]
#     V = rays.shape[0]
#     # expand rays to have the same dimension as desdf
#     rays = rays.reshape((1, 1, -1))
# 
#     # circular pad the desdf
#     pad_front = V // 2
#     pad_back = V - pad_front
#     pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")
# 
#     # probablility is -l1norm
#     prob_vol = torch.stack(
#         [
#             -torch.norm(pad_desdf[:, :, i : i + V] - rays, p=1.0, dim=2)
#             for i in range(O)
#         ],
#         dim=2,
#     )  # (H,W,O)
#     prob_vol = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive
# 
#     # maxpooling
#     prob_dist, orientations = torch.max(prob_vol, dim=2)
# 
#     # get the prediction
#     #pred_y, pred_x = np.unravel_index(np.argmax(prob_dist.numpy()), prob_dist.shape)
#     prob_dist_cpu = prob_dist.cpu().numpy() # Move the tensor to the CPU and convert to numpy
#     pred_y, pred_x = np.unravel_index(np.argmax(prob_dist_cpu), prob_dist_cpu.shape)
#     pred_x = torch.from_numpy(np.array([pred_x]))
#     pred_y = torch.from_numpy(np.array([pred_y]))
#     #pred_y, pred_x = torch.where(prob_dist == prob_dist.max())
#     orn = orientations[pred_y, pred_x]
#     # from orientation indices to radians
#     orn = orn / orn_slice * 2 * torch.pi
#     pred = torch.cat((pred_x, pred_y, orn))
#     if return_np:
#         return (
#             prob_vol.detach().cpu().numpy(),
#             prob_dist.detach().cpu().numpy(),
#             orientations.detach().cpu().numpy(),
#             pred.detach().cpu().numpy(),
#         )
#     else:
#         return (
#             prob_vol.to(torch.float32).detach().cpu(),
#             prob_dist.to(torch.float32).detach().cpu(),
#             orientations.to(torch.float32).detach().cpu(),
#             pred.to(torch.float32).detach().cpu(),
#         )


def get_ray_from_depth(d, V=11, dv=10, a0=None, F_W=3 / 8):
    """
    Shoot the rays to the depths, from left to right
    Input:
        d: 1d depths from image
        V: number of rays
        dv: angle between two neighboring rays
        a0: camera intrisic
        F/W: focal length / image width
    Output:
        rays: interpolated rays
    """
    W = d.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi

    if a0 is None:
        # assume a0 is in the middle of the image
        w = np.tan(angles) * W * F_W + (W - 1) / 2  # desired width, left to right
    else:
        w = np.tan(angles) * W * F_W + a0  # left to right

    interp_d = griddata(np.arange(W).reshape(-1, 1), d, w, method="linear")
    rays = interp_d / np.cos(angles)

#    print(f"rays shape: {rays.shape}")
#    print(f"rays range: {rays.min()} to {rays.max()}")

    return rays




def get_ray_from_depth_uncertainty(d, scales, V=11, dv=10, a0=None, F_W=3 / 8):
    """
    Shoot the rays to the depths, from left to right
    Input:
        d: 1d depths from image
        V: number of rays
        dv: angle between two neighboring rays
        a0: camera intrisic
        F/W: focal length / image width
    Output:
        rays: interpolated rays
    """
    W = d.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi

    if a0 is None:
        # assume a0 is in the middle of the image
        w = np.tan(angles) * W * F_W + (W - 1) / 2  # desired width, left to right
    else:
        w = np.tan(angles) * W * F_W + a0  # left to right

    interp_d = griddata(np.arange(W).reshape(-1, 1), d, w, method="linear")
    rays = interp_d / np.cos(angles)
    scales_rays = griddata(np.arange(W).reshape(-1, 1), scales, w, method="linear")

    print(f"rays shape: {rays.shape}")
    print(f"rays range: {rays.min()} to {rays.max()}")

    return rays, scales_rays


def get_ray_from_depth_uncertainty_semantics(d, scales, semantics, V=11, dv=10, a0=None, F_W=3 / 8):
    """
    Shoot the rays to the depths, from left to right
    Input:
        d: 1d depths from image
        V: number of rays
        dv: angle between two neighboring rays
        a0: camera intrisic
        F/W: focal length / image width
    Output:
        rays: interpolated rays
    """
    W = d.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi

    if a0 is None:
        # assume a0 is in the middle of the image
        w = np.tan(angles) * W * F_W + (W - 1) / 2  # desired width, left to right
    else:
        w = np.tan(angles) * W * F_W + a0  # left to right

    interp_d = griddata(np.arange(W).reshape(-1, 1), d, w, method="linear")
    rays = interp_d / np.cos(angles)
    scales_rays = griddata(np.arange(W).reshape(-1, 1), scales, w, method="linear")
    semantics_rays = griddata(np.arange(W).reshape(-1, 1), semantics, w, method="linear")

    print(f"rays shape: {rays.shape}")
    print(f"rays range: {rays.min()} to {rays.max()}")

    return rays, scales_rays, semantics_rays



def transit(
    prob_vol,
    transition,
    sig_o=0.1,
    sig_x=0.05,
    sig_y=0.05,
    tsize=5,
    rsize=5,
    resolution=0.1,
):
    """
    Input:
        prob_vol: torch.tensor(H, W, O), probability volume before the transition
        transition: ego motion
        sig_o: stddev of rotation
        sig_x: stddev in x translation
        sig_y: stddev in y translation
        tsize: translational filter size
        rsize: rotational filter size
        resolution: resolution of the grid [m/pixel]
    """
    H, W, O = list(prob_vol.shape)
    # construction O filters
    filters_trans, filter_rot = get_filters(
        transition,
        O,
        sig_o=sig_o,
        sig_x=sig_x,
        sig_y=sig_y,
        tsize=tsize,
        rsize=rsize,
        resolution=resolution,
    )  # (O, 5, 5), (5,)

    if torch.isnan(prob_vol).any():
        print("NaNs found in prob_vol1")

    ### Added 01.07.2024
    # Move filters to the same device as prob_vol
    device = prob_vol.device
    filters_trans = filters_trans.to(device)
    filter_rot = filter_rot.to(device)
    ###

    # set grouped 2d convolution, O as channels
    prob_vol = prob_vol.permute((2, 0, 1))  # (O, H, W)
    #print("prob_vol2: ", prob_vol)
    if torch.isnan(prob_vol).any():
        print("NaNs found in prob_vol2")
    # convolve with the translational filters
    # NOTE: make sure the filter is convolved correctly need to flip
    prob_vol = F.conv2d(
        prob_vol,
        weight=filters_trans.unsqueeze(1).flip([-2, -1]),
        bias=None,
        groups=O,
        padding="same",
    )  # (O, H, W)
    #print("prob_vol3: ", prob_vol)
    if torch.isnan(prob_vol).any():
        print("NaNs found in prob_vol3")
    # convolve with rotational filters
    # reshape as batch
    prob_vol = prob_vol.permute((1, 2, 0))  # (H, W, O)
    prob_vol = prob_vol.reshape((H * W, 1, O))  # (HxW, 1, O)
    prob_vol = F.pad(
        prob_vol, pad=[int((rsize - 1) / 2), int((rsize - 1) / 2)], mode="circular"
    )
    #print("prob_vol3a: ", prob_vol)
    if torch.isnan(prob_vol).any():
        print("3a: NaNs found in prob_vol")
    

    prob_vol = F.conv1d(
        prob_vol, weight=filter_rot.flip(dims=[-1]).unsqueeze(0).unsqueeze(0), bias=None
    )  # TODO (HxW, 1, O)
    if torch.isnan(prob_vol).any():
        print("4: NaNs found in prob_vol")
    # print("prob_vol4: ", prob_vol)
    # reshape
    prob_vol = prob_vol.reshape([H, W, O])  # (H, W, O)
    # normalize
    epsilon = 1e-7
    prob_vol = prob_vol / (prob_vol.sum() + epsilon)
    # print("prob_vol5: ", prob_vol)
    if torch.isnan(prob_vol).any():
        print("5: NaNs found in prob_vol")

    print(f"filters_trans shape: {filters_trans.shape}")
    print(f"filters_trans range: {filters_trans.min().item()} to {filters_trans.max().item()}")

    print(f"filter_rot shape: {filter_rot.shape}")
    print(f"filter_rot range: {filter_rot.min().item()} to {filter_rot.max().item()}")

    print(f"prob_vol shape: {prob_vol.shape}")
    print(f"prob_vol range: {prob_vol.min().item()} to {prob_vol.max().item()}")

    return prob_vol


def get_filters(
    transition,
    O=36,
    sig_o=0.1,
    sig_x=0.05,
    sig_y=0.05,
    tsize=5,
    rsize=5,
    resolution=0.1,
):
    """
    Return O different filters according to the ego-motion
    Input:
        transition: torch.tensor (3,), ego motion
    Output:
        filters_trans: torch.tensor (O, 5, 5)
                    each filter is (fH, fW)
        filters_rot: torch.tensor (5)
    """
    # NOTE: be careful about the orienation order, what is the orientation of the first layer?

    # get the filters according to gaussian
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
    )
    # add units
    grid_x = grid_x * resolution  # 0.1m
    grid_y = grid_y * resolution  # 0.1m

    # calculate center of the gaussian for 36 orientations
    # center for orientation stays the same
    center_o = transition[-1]
    # center_x and center_y depends on the orientation, in total O different, rotate
    orns = (
        torch.arange(0, O, dtype=torch.float32, device=transition.device)
        / O
        * 2
        * torch.pi
    )  # (O,)
    c_th = torch.cos(orns).reshape((O, 1, 1))  # (O, 1, 1)
    s_th = torch.sin(orns).reshape((O, 1, 1))  # (O, 1, 1)
    center_x = transition[0] * c_th - transition[1] * s_th  # (O, 1, 1)
    center_y = transition[0] * s_th + transition[1] * c_th  # (O, 1, 1)
#    center_y = transition[0] * c_th - transition[1] * s_th  # (O, 1, 1) # try
#    center_x = transition[0] * s_th + transition[1] * c_th  # (O, 1, 1)

    # add uncertainty
    filters_trans = torch.exp(
        -((grid_x - center_x) ** 2) / (sig_x**2) - (grid_y - center_y) ** 2 / (sig_y**2)
    )  # (O, 5, 5)

    if torch.isnan(filters_trans).any():
        print("NaNs found in filters_trans2")

    # normalize
    epsilon = 1e-10
    filters_trans = filters_trans / (filters_trans.sum(-1).sum(-1).reshape((O, 1, 1)) + epsilon)
    #filters_trans = filters_trans / filters_trans.sum(-1).sum(-1).reshape((O, 1, 1))
    if torch.isnan(filters_trans).any():
        print("NaNs found in filters_trans3")


    # rotation filter
    grid_o = (
        torch.arange(-(rsize - 1) / 2, (rsize + 1) / 2, 1, device=transition.device)
        / O
        * 2
        * torch.pi
    )
    filter_rot = torch.exp(-((grid_o - center_o) ** 2) / (sig_o**2))  # (5)

    if torch.isnan(filter_rot).any():
        print("NaNs found in filter_rot")

    return filters_trans, filter_rot
