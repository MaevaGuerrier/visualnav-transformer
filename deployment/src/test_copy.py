import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# VIZ_IMAGE_SIZE = (640, 480)

def project_points(
    xy: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    camera_height: float = 0.25,
    camera_x_offset: float = 0.10,
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




if __name__ == "__main__":
    read_image = cv2.imread("../debug/obs_img.png")
    naction = [[
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
    ]]

CAM_FORWARD = 0.00   # +x_r (forward)
CAM_LEFT    = 0.10   # +y_r (left)   <-- set this if your 10 cm was lateral
CAM_UP      = 0.25   # +z_r (up)

# ---- Fisheye intrinsics from your calibration ----
camera_matrix = np.array([[262.459286,   1.916160, 327.699961],
              [  0.000000, 263.419908, 224.459372],
              [  0.000000,   0.000000,   1.000000]], dtype=np.float64)

# Distortion (k1, k2, k3, k4)
dist_coeffs = np.array([-0.03727222045233312, 0.007588870705292973,
              -0.01666117486022043, 0.00581938967971292], dtype=np.float64)



naction = np.array(naction)


fig, ax = plt.subplots()

ax.imshow(read_image)

for i, traj in enumerate(naction):
    xy_coords = traj[:, :2]  # (horizon, 2)

   
    traj_pixels = project_points(
        xy_coords[np.newaxis], camera_matrix, dist_coeffs
    )[0]
    # traj_pixels[:, 0] = VIZ_IMAGE_SIZE[0] - traj_pixels[:, 0]

    ax.plot(
        traj_pixels[:250, 0],
        traj_pixels[:250, 1],
        lw=2.5,
    )


ax.figure.savefig('../debug_viz/img_test2.png')
