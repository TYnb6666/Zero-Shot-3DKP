#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from .graphics_utils import fov2focal, getWorld2View2, getProjectionMatrix
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix

def colmap_to_pytorch3d(rotation,translation,device):
    '''
    Changes the rotation and translation from the colmap to the pytorch 3D convention.
    '''
    rotation = torch.stack([-rotation[:, 0], -rotation[:, 1], rotation[:, 2]], 1) # from RDF to Left-Up-Forward for Rotation
    new_c2w = torch.cat([rotation, translation], 1)
    bottom = torch.Tensor([[0,0,0,1]]).to(device)
    w2c = torch.linalg.inv(torch.cat((new_c2w, bottom), 0))
    rotation, translation = w2c[:3, :3].permute(1, 0), w2c[:3, 3]

    return rotation, translation

def convert_camera_from_gs_to_pytorch3d(gs_cameras, device='cuda'):
    """
    From Gaussian Splatting camera parameters,
    computes R, T, K matrices and outputs pytorch3d-compatible camera object.

    Args:
        gs_cameras (List of GSCamera): List of Gaussian Splatting cameras.
        device (_type_, optional): _description_. Defaults to 'cuda'.

    Returns:
        p3d_cameras: pytorch3d-compatible camera object.
    """
    
    N = len(gs_cameras)
    
    # TODO: Directly use the torch matrix, without converting into array.
    R = torch.from_numpy(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device, dtype=torch.float32)
    T = torch.from_numpy(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device, dtype=torch.float32)
    fx = torch.from_numpy(np.array([fov2focal(gs_camera.FovX, gs_camera.width) for gs_camera in gs_cameras])).to(device, dtype=torch.float32)
    fy = torch.from_numpy(np.array([fov2focal(gs_camera.FovY, gs_camera.height) for gs_camera in gs_cameras])).to(device, dtype=torch.float32)
    image_height = torch.from_numpy(np.array([gs_camera.height for gs_camera in gs_cameras])).to(device, dtype=torch.float32)
    image_width = torch.from_numpy(np.array([gs_camera.width for gs_camera in gs_cameras])).to(device, dtype=torch.float32)
    cx = image_width / 2.  # torch.zeros_like(fx).to(device)
    cy = image_height / 2.  # torch.zeros_like(fy).to(device)
    
    w2c = torch.zeros(N, 4, 4).to(device)
    w2c[:, :3, :3] = R.transpose(-1, -2)
    w2c[:, :3, 3] = T
    w2c[:, 3, 3] = 1
    
    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1
    c2w = c2w[:, :3, :]
    
    distortion_params = torch.zeros(N, 6).to(device)
    camera_type = torch.ones(N, 1, dtype=torch.int32).to(device)

    # Pytorch3d-compatible camera matrices
    # Intrinsics
    image_size = torch.tensor(
        [image_width[0], image_height[0]], device=device, dtype=torch.float32
    ).unsqueeze_(0)
    # image_size = torch.cat([image_width.view(-1, 1), image_height.view(-1, 1)], dim=-1)
    
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    
    # p0_pytorch3d = (
    #     -(
    #         torch.Tensor(
    #             (cx[0], cy[0]),
    #         )[
    #             None
    #         ].to(device)
    #         - c0
    #     )
    #     / scale
    # )
    p0_pytorch3d = -(torch.cat([cx.view(-1, 1), cy.view(-1, 1)], dim=-1) - c0) / scale
    
    # focal_pytorch3d = (
    #     torch.Tensor([fx[0], fy[0]])[None].to(device) / scale
    # )
    focal_pytorch3d = torch.cat([fx.view(-1, 1), fy.view(-1, 1)], dim=-1) / scale
    
    # print("focalp3d, p0p3d:", focal_pytorch3d.shape, p0_pytorch3d.shape)

    K = _get_sfm_calibration_matrix(
        N, "cpu", focal_pytorch3d, p0_pytorch3d, orthographic=False
    )
    # print("K:", K.shape)
    # K = K.expand(N, -1, -1)
    if K.shape[0] != N:
        raise ValueError("K shape does not match the number of cameras.")

    # Extrinsics
    line = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32).expand(N, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    R, T = world2cam.split([3, 1], dim=-1)
    R = R[:, :3].transpose(1, 2) * torch.tensor([-1.0, 1.0, -1], device=device, dtype=torch.float32)
    T = T.squeeze(2)[:, :3] * torch.tensor([-1.0, 1.0, -1], device=device, dtype=torch.float32)

    p3d_cameras = FoVPerspectiveCameras(device=device, R=R, T=T, K=K, znear=0.0001)

    return p3d_cameras


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.gt_mask = None

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
