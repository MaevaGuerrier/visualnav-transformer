# from vint_utils import plot_trajs_and_points_on_image
# import cv2
# import numpy as np
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path


# if __name__ == "__main__":
#     read_image = cv2.imread("../debug/obs_img.png")
traj = [[
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

#     print(traj)

#     plot_trajs_and_points_on_image(
#         img=read_image,
#         list_trajs=traj,
#         pub=True,
#     )
    



# ---- User inputs ----
IMAGE_PATH = "../debug/obs_img.png"   # <-- change to your image
# A single 2D waypoint in robot frame, meters (x_r forward, y_r left); z_r assumed 0
WAYPOINT_R = (1.0, 0.5)

# Camera pose relative to robot CoG (robot frame, meters):
CAM_FORWARD = 0.00   # +x_r (forward)
CAM_LEFT    = 0.10   # +y_r (left)   <-- set this if your 10 cm was lateral
CAM_UP      = 0.25   # +z_r (up)

# ---- Fisheye intrinsics from your calibration ----
K = np.array([[262.459286,   1.916160, 327.699961],
              [  0.000000, 263.419908, 224.459372],
              [  0.000000,   0.000000,   1.000000]], dtype=np.float64)

# Distortion (k1, k2, k3, k4)
D = np.array([-0.03727222045233312, 0.007588870705292973,
              -0.01666117486022043, 0.00581938967971292], dtype=np.float64)

# ---- Robot->Camera rotation (forward-facing) ----
# Maps robot axes (+x forward, +y left, +z up) to camera axes (+x right, +y down, +z forward)
R = np.array([[0.0, -1.0,  0.0],
              [0.0,  0.0, -1.0],
              [1.0,  0.0,  0.0]], dtype=np.float64)

# Camera position in robot frame
C_r = np.array([[CAM_FORWARD],
                [CAM_LEFT],
                [CAM_UP]], dtype=np.float64)

# OpenCV expects: X_c = R * X_r + t  where t = -R * C_r
t = -R @ C_r  # 3x1

cam_translation = np.array([[CAM_FORWARD], [0.0], [CAM_UP]], dtype=np.float64)

def project_robot_point_to_undistorted(img, point_robot_xy, K, D, R, t):
    """
    point_robot_xy: (x_r, y_r) in meters, z_r assumed 0.
    Returns: undistorted_img, undist_pt (u,v) or None if not visible.
    """
    if img is None:
        raise FileNotFoundError("Could not read IMAGE_PATH.")

    h, w = img.shape[:2]

    # Build 3D point in robot frame (on ground plane)
    Xr = np.array([[point_robot_xy[0], point_robot_xy[1], 0.0]], dtype=np.float64)  # (1,3)
    Xc = (R @ Xr.T + t).T  # (1,3)

    # Visibility check: camera sees points with Z_c > 0
    Zc = Xc[0, 2]
    if Zc <= 0:
        print(f"[WARN] Point has Z_c={Zc:.3f} (<=0): not in front of camera. Returning None.")
        # Still produce undistorted image for debugging
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1.0, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, np.eye(3), newK, (w, h), cv2.CV_32FC1)
        undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        return undist, None

    # Project to distorted fisheye pixels
    obj = Xc.reshape(-1, 1, 3)  # (1,1,3)
    # rvec, tvec are zero because obj is already in camera frame
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    pix_distorted, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
    u_d, v_d = pix_distorted.ravel()

    # Undistort the image and remap the point into undistorted coordinates
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1.0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, np.eye(3), newK, (w, h), cv2.CV_32FC1)
    undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Map the single distorted pixel to undistorted pixel using the same maps
    u_i, v_i = int(round(u_d)), int(round(v_d))
    if 0 <= u_i < w and 0 <= v_i < h:
        u_u = mapx[v_i, u_i]
        v_u = mapy[v_i, u_i]
        undist_pt = (int(round(float(u_u))), int(round(float(v_u))))
        # Draw
        cv2.circle(undist, undist_pt, 6, (0, 0, 255), -1)
    else:
        print("[WARN] Projected distorted pixel lies outside image. Returning None.")
        undist_pt = None

    return undist, undist_pt


def project_points(points_2d_robot, K, D, R, t):
    """Project robot-frame 2D ground points into image coordinates."""
    # Convert to 3D points (x forward, y left, z = 0 ground plane)
    pts_robot = np.array([[p[0], p[1], 0.0] for p in points_2d_robot], dtype=np.float32).T

    # Transform to camera frame
    pts_cam = R @ pts_robot + t

    # OpenCV expects shape (N,1,3)
    pts_cam = pts_cam.T.reshape(-1, 1, 3)

    # Project with fisheye model
    pts_img, _ = cv2.fisheye.projectPoints(pts_cam, np.zeros((3, 1)), np.zeros((3, 1)), K, D)

    return pts_img.reshape(-1, 2)






def main():
    img = cv2.imread(IMAGE_PATH)
    undist, pt = project_robot_point_to_undistorted(img, WAYPOINT_R, K, D, R, t)

    print("Projected point (undistorted):", pt)
    out_path = Path("../debug_viz/undistorted_with_point.png")
    cv2.imwrite(str(out_path), undist)
    print(f"Saved: {out_path.resolve()}")


    # ==== PROJECT AND DRAW ====
    colors = [(0, 255, 0), (0, 0, 255)]  # green, red for trajectories

    for idx, wp_list in enumerate(traj):
        projected = project_points(wp_list, K, D, R, cam_translation)

        # Draw polyline
        pts_int = np.int32(projected)
        cv2.polylines(img, [pts_int], isClosed=False, color=colors[idx % len(colors)], thickness=2)

        # Draw waypoints
        for p in pts_int:
            cv2.circle(img, tuple(p), 4, (255, 255, 255), -1)

    # ==== SAVE RESULT ====
    cv2.imwrite("../debug_viz/undistorted_with_traj.png", img)
    print("Saved image with trajectories -> undistorted_with_traj.png")




if __name__ == "__main__":
    main()
