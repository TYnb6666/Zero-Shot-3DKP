import numpy as np
import pyrr


#####################################################################
####################### CAMERA PARAMETERS     #######################
#####################################################################
def get_camera_parameters(elevation_deg, azimuth_deg):
    # Convert angles to radians
    elevation_rad = np.radians(elevation_deg)
    azimuth_rad = np.radians(azimuth_deg)

    # Camera position and target point
    camera_pos = np.array(
        [
            2 * np.cos(elevation_rad) * np.sin(azimuth_rad),
            2 * np.cos(elevation_rad) * np.cos(azimuth_rad),
            2 * np.sin(elevation_rad),
        ]
    )

    target = np.array([0, 0, 0])  # Camera always looks at the origin

    # Define the camera's up vector (assuming it's pointing upwards)
    up = np.array([0, 0, 1])

    # Calculate the camera's forward, right, and up vectors
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # Create the view matrix
    view_matrix = np.eye(4)
    view_matrix[:3, 0] = right
    view_matrix[:3, 1] = up
    view_matrix[:3, 2] = -forward
    view_matrix[:3, 3] = -camera_pos

    return camera_pos, up, target


# Default target Looking at the origin
def get_camera_poses(target=np.zeros(3),
                     radius=2,
                     elevation_deg=30,
                     azimuth_deg_range=(0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330),
                     num_random_views=10,
                     std=4,
                     center_elev=np.pi,
                     center_azim=0.0,
                     seed=2024,
                     add_random_views=False):
    camera_poses = []
    elevation_rad = np.radians(elevation_deg)
    up = np.array([0.0, 1.0, 0.0])  # Y-axis is up

    for azimuth_deg in azimuth_deg_range:
        azimuth_rad = np.radians(azimuth_deg)

        # Convert spherical coordinates to Cartesian coordinates
        x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = radius * np.sin(elevation_rad)
        z = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)

        eye = np.array([x, y, z])

        # Create camera pose
        camera_pose = np.array(
            pyrr.Matrix44.look_at(
                eye=eye, target=target, up=up
            ).T
        )
        camera_pose = np.linalg.inv(camera_pose)
        camera_poses.append(camera_pose)

    # Random views
    if add_random_views:
        # Initialize random state
        rand = np.random.RandomState(seed)

        random_elevs = rand.randn(num_random_views) * np.pi / std + center_elev
        random_azims = rand.randn(num_random_views) * 2 * np.pi / std + center_azim

        for elev, azim in zip(random_elevs, random_azims):
            x = radius * np.cos(elev) * np.cos(azim)
            y = radius * np.sin(elev)
            z = radius * np.cos(elev) * np.sin(azim)

            eye = np.array([x, y, z])

            camera_pose = np.array(
                pyrr.Matrix44.look_at(
                    eye=eye, target=target, up=up
                ).T
            )
            camera_pose = np.linalg.inv(camera_pose)
            camera_poses.append(camera_pose)

    return np.stack(camera_poses, 0)
