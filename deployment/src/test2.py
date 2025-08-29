import cv2
import numpy as np
import matplotlib.pyplot as plt

VIZ_IMAGE_SIZE = (640, 480)

def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.

    Args:
        xy: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients


    Returns:
        uv: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = (0, 0, 0)

    xyz[..., 0] += camera_x_offset
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)

    return uv

def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    clip: bool = False,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        points: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients

    Returns:
        pixels: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, VIZ_IMAGE_SIZE[0]),
                    np.clip(p[1], 0, VIZ_IMAGE_SIZE[1]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
            ]
        )
    return pixels




def plot_trajs_and_points_on_image(
    ax: plt.Axes,
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    list_trajs: list,
):
    """
    Plot trajectories and points on an image. If there is no configuration for the camera interinstics of the dataset, the image will be plotted as is.
    Args:
        ax: matplotlib axis
        img: image to plot
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
    """
    ax.imshow(img)

    camera_height = 0.25
    camera_x_offset = 0.10


    for i, traj in enumerate(list_trajs):
        xy_coords = traj[:, :2]  # (horizon, 2)
        traj_pixels = get_pos_pixels(
            xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=False
        )
        if len(traj_pixels.shape) == 2:
            ax.plot(
                traj_pixels[:250, 0],
                traj_pixels[:250, 1],
                lw=2.5,
            )

    # for i, point in enumerate(list_points):
    #     if len(point.shape) == 1:
    #         # add a dimension to the front of point
    #         point = point[None, :2]
    #     else:
    #         point = point[:, :2]
    #     pt_pixels = get_pos_pixels(
    #         point, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=True
    #     )
    #     ax.plot(
    #         pt_pixels[:250, 0],
    #         pt_pixels[:250, 1],
    #         color=point_colors[i],
    #         marker="o",
    #         markersize=10.0,
    #     )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
    ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))











if __name__ == "__main__":
    # Load or create test image
    try:
        read_image = cv2.imread("../debug/obs_img.png")
        if read_image is None:
            raise FileNotFoundError
    except:
        print("ISSUE WHEN READING IMG..")

    # Sample trajectories
    trajs = np.array([[
        [0.04970074,  0.01101923],
        [0.10549255,  0.01843739],
        [0.16880728, -0.01180482],
        [0.24336472, -0.05049229],
        [0.29995292, -0.08638072],
        [0.32483295, -0.09744167],
        [0.31955287, -0.08121347],
        [0.27909696, -0.0196743]
    ],
    [
        [0.02367929,  0.01485968],
        [0.06050363,  0.01524925],
        [0.10216802,  0.0234971],
        [0.14909938,  0.04729271],
        [0.16844198,  0.04669142],
        [0.14706865,  0.07573462],
        [0.09317756,  0.13997173],
        [0.02677932,  0.25704145]
    ]])

    camera_matrix = np.array([
        [262.459286,   1.916160, 327.699961],
        [  0.000000, 263.419908, 224.459372],
        [  0.000000,   0.000000,   1.000000]
    ], dtype=np.float64)

    dist_coeffs = np.array([
        -0.03727222045233312, 
         0.007588870705292973,
        -0.01666117486022043, 
         0.00581938967971292
    ], dtype=np.float64)

    # Test different Y-axis corrections
    fig, ax = plt.subplots()


    # [start_pos, goal_pos],
    plot_trajs_and_points_on_image(
        ax,
        read_image,
        camera_matrix,
        dist_coeffs,
        trajs
    )

    fig.savefig('../debug_viz/img_test2.png')