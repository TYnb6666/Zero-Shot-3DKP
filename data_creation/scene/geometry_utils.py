import math
import torch
from typing import List
from tqdm import tqdm

from .general_utils import build_rotation
from .cameras import Camera

def find_nan_values(tensor, name: str = ""):
    if torch.isnan(tensor).any():
        nan_mask = torch.isnan(tensor)
        nan_locations = torch.nonzero(nan_mask)
        print(f"[WARNING] Found {nan_locations.shape[0]} NaN values in {name}")
        print(f"NaN locations (channel, y, x):")
        for loc in nan_locations[:10]:  # Print first 10 locations
            print(f"  {loc.tolist()}")
        if nan_locations.shape[0] > 10:
            print("  ...")


def get_gaussian_normals_from_view(view, gaussians, in_view_space=True):
    # Build rotation matrices of Gaussians
    gaussian_rots = build_rotation(gaussians._rotation)

    # Get the minimum scale index for each Gaussian
    gaussian_min_scale_idx = gaussians.get_scaling_with_3D_filter.min(dim=-1)[1][:, None, None].repeat(1, 3, 1)

    # Gather the normals as the shortest axis of the covariance matrices
    gaussian_normals = torch.gather(gaussian_rots, dim=2, index=gaussian_min_scale_idx).squeeze()

    # Flip the normals if they are pointing away from the camera
    gaussian_normals = gaussian_normals * torch.sign(
        (
            gaussian_normals * (view.camera_center[None] - gaussians.get_xyz)
        ).sum(dim=-1, keepdim=True)
    )

    if in_view_space:
        gaussian_normals = (gaussian_normals @ view.world_view_transform[:3,:3])

    return gaussian_normals


def transform_points_world_to_view(
    points:torch.Tensor,
    cameras:List[Camera],
    use_p3d_convention:bool=False,
):
    """Transform points from world space to view space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    world_view_transforms = torch.stack([camera.world_view_transform for camera in cameras], dim=0)  # (n_cameras, 4, 4)

    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    view_points = (points_h @ world_view_transforms)[..., :3]  # (n_cameras, N, 3)
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        view_points = factors * view_points  # (n_cameras, N, 3)
    return view_points


def transform_points_view_to_world(
    points:torch.Tensor,
    cameras:List[Camera],
    use_p3d_convention:bool=False,
):
    """Transform points from view space to world space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    view_world_transforms = torch.stack([camera.world_view_transform.inverse() for camera in cameras], dim=0)  # (n_cameras, 4, 4)

    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        points = factors * points  # (n_cameras, N, 3)
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    world_points = (points_h @ view_world_transforms)[..., :3]  # (n_cameras, N, 3)
    return world_points


def transform_points_to_pixel_space(
        points:torch.Tensor,
        cameras:List[Camera],
        points_are_already_in_view_space:bool=False,
        use_p3d_convention:bool=False,
        znear:float=1e-6,
        keep_float:bool=False,
    ):
        """Transform points from world space (3 coordinates) to pixel space (2 coordinates).

        Args:
            points (torch.Tensor): Should have shape (n_cameras, N, 3).
            cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
            points_are_already_in_view_space (bool, optional): Defaults to False.
            use_p3d_convention (bool, optional): Defaults to False.
            znear (float, optional): Defaults to 1e-6.

        Returns:
            torch.Tensor: Has shape (n_cameras, N, 2).
                In pixel space, (0, 0) is the center of the left-top pixel,
                and (W-1, H-1) is the center of the right-bottom pixel.
        """
        if points_are_already_in_view_space:
            full_proj_transforms = torch.stack([camera.projection_matrix for camera in cameras])  # (n_depth, 4, 4)
            if use_p3d_convention:
                points = torch.tensor([[[-1, -1, 1]]], device=points.device) * points
        else:
            full_proj_transforms = torch.stack([camera.full_proj_transform for camera in cameras])  # (n_cameras, 4, 4)

        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
        proj_points = points_h @ full_proj_transforms  # (n_cameras, N, 4)
        proj_points = proj_points[..., :2] / proj_points[..., 3:4].clamp_min(znear)  # (n_cameras, N, 2)
        # proj_points is currently in a normalized space where
        # (-1, -1) is the left-top corner of the left-top pixel,
        # and (1, 1) is the right-bottom corner of the right-bottom pixel.

        # For converting to pixel space, we need to scale and shift the normalized coordinates
        # such that (-1/2, -1/2) is the left-top corner of the left-top pixel,
        # and (H-1/2, W-1/2) is the right-bottom corner of the right-bottom pixel.

        height, width = cameras[0].image_height, cameras[0].image_width
        image_size = torch.tensor([[width, height]], device=points.device)

        # proj_points = (1. + proj_points) * image_size / 2
        proj_points = (1. + proj_points) / 2 * image_size - 1./2.

        if keep_float:
            return proj_points
        else:
            return torch.round(proj_points).long()

def get_camera_rays(view: Camera):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().cuda()
    rays_d = intrins_inv @ points
    return rays_d

# the following functions are adopted from RaDe-GS:
def depths_to_points(view, depthmap1, depthmap2=None, return_rays_d=False):
    W, H = view.image_width, view.image_height
    rays_d = get_camera_rays(view)
    points1 = depthmap1.reshape(1,-1) * rays_d

    pckg = (points1.reshape(3,H,W),)
    if depthmap2 is not None:
        points2 = depthmap2.reshape(1,-1) * rays_d
        pckg += (points2.reshape(3,H,W),)
    if return_rays_d:
        pckg += (rays_d,)

    return pckg


def point_to_normal(view, points1, points2=None):
    points = (
        points1[None] if points2 is None
        else torch.stack([points1, points2],dim=0)
    )
    output = torch.zeros_like(points)
    dx = points[...,2:, 1:-1] - points[...,:-2, 1:-1]
    dy = points[...,1:-1, 2:] - points[...,1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=1), dim=1)
    output[...,1:-1, 1:-1] = normal_map
    return (
        output[0] if points2 is None
        else output
    )


def depth_to_normal(view, depth1, depth2=None):
    points = depths_to_points(view, depth1, depth2)
    points = points[None] if depth2 is None else points  # type: ignore[call-overload, arg-type]
    return point_to_normal(view, *points)


def is_in_view_frustum(
    points:torch.Tensor,
    camera:Camera,
) -> torch.Tensor:
    """_summary_

    Args:
        points (torch.Tensor): Tensor with shape (N, 3)
        cameras (List[Camera]): _description_
    """
    H, W = camera.image_height, camera.image_width

    view_points = transform_points_world_to_view(
        points.view(1, -1, 3),
        cameras=[camera],
    )[0]  # (N, 3)

    pix_pts = transform_points_to_pixel_space(
        view_points.view(1, -1, 3),
        points_are_already_in_view_space=True,
        cameras=[camera],
    )[0]  # (N, 2)

    pix_x, pix_y, pix_z = pix_pts[..., 0], pix_pts[..., 1], view_points[..., 2]

    valid_mask = (
        (pix_x >= 0) & (pix_x <= W-1)
        & (pix_y >= 0) & (pix_y <= H-1)
        & (pix_z > camera.znear) & (pix_z < camera.zfar)
    )  # (N,)

    return valid_mask


def map_depth_and_normal_to_color(depth_map: torch.Tensor,
                       normal_map: torch.Tensor,
                       color_field: torch.nn.Module,
                       camera_view: torch.Tensor,
                       epsilon: float = 1e-6,
                       show_progress: bool = False,
                       chunk_size: int = 500_000,
                       max_random_points: int = 1_000_000) -> torch.Tensor:
    """
    Converts a depth map into world space coordinates and retrieves the corresponding color from a color field model.

    Args:
        depth_map (torch.Tensor): Tensor representing the depth map.
        color_field_model (torch.nn.Module): Model that maps world coordinates and ray directions to colors.
        camera_view (torch.Tensor): Tensor representing the camera viewpoint.
        show_progress (bool, optional): Flag to enable progress bar display. Defaults to False.
        NOTE: The function can be heavy on memory during training, so we add chunkify and random sampling mechanisms.
        chunk_size (int, optional): Number of points to process in each chunk. Defaults to 500,000.
        max_random_points (int, optional): Maximum number of random points to sample. Defaults to 1,000,000.

    Returns:
        torch.Tensor: Tensor containing the color information corresponding to the input depth map.
    """
    # Transform depth to world space
    world_coords, ray_directions = depths_to_points(view=camera_view, depthmap1=depth_map, depthmap2=None, return_rays_d=True)  # type: ignore[misc]
    world_coords = world_coords.view(3, -1).permute(1, 0).unsqueeze(0)
    world_coords = transform_points_view_to_world(world_coords, [camera_view]).squeeze(0)  # type: ignore[arg-type]

    ray_directions = ray_directions.permute(1, 0).unsqueeze(0) # ray_directions are in view space
    image_plane_coords = transform_points_view_to_world(ray_directions, [camera_view]).squeeze(0) # we retrieve the image plane in world space  # type: ignore[arg-type]
    view_directions = world_coords - image_plane_coords
    view_directions_norm = view_directions.norm(dim=-1, keepdim=True)
    view_directions_norm[view_directions_norm == 0] = epsilon
    view_directions = view_directions / view_directions_norm # Normalize ray directions


    color_output = torch.zeros((world_coords.size(0), 3), device=world_coords.device)  # Initialize a zero tensor

    random_mask = None
    if max_random_points > 0:
        # Create random mask
        random_mask = torch.randperm(world_coords.size(0))[:max_random_points]
        world_coords = world_coords[random_mask]
        view_directions = view_directions[random_mask]

    if show_progress:
        print(f"[INFO] Processing color field with chunk size: {chunk_size / 1_000_000}M with shape: {world_coords.shape[0] / 1_000_000}M")
    for i in tqdm(range(0, world_coords.size(0), chunk_size), desc="Processing color field", disable=not show_progress):
        coords_chunk = world_coords[i:i + chunk_size]
        directions_chunk = view_directions[i:i + chunk_size]
        color_chunk = color_field(coords_chunk, view_directions=directions_chunk)
        if random_mask is None:
            color_output[i:i + chunk_size] = color_chunk
        else:
            color_output[random_mask[i:i + chunk_size]] = color_chunk
    color_output = color_output.permute(1, 0)
    return color_output.view(3, camera_view.image_height, camera_view.image_width)  # type: ignore[attr-defined]