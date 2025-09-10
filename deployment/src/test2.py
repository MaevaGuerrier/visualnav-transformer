import cv2
import numpy as np
import matplotlib.pyplot as plt

from cv_bridge import CvBridge

VIZ_IMAGE_SIZE = (640, 480)

bridge = CvBridge()

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
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = np.zeros((3, 1), dtype=np.float64)

    xyz[..., 0] += camera_x_offset

    # Convert from (x, y, z) to (y, -z, x) for cv2
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    
    # done for cv2.fisheye.projectPoint requires float32/float64 and shape (N,1,3),
    xyz_cv = xyz_cv.reshape(batch_size * horizon, 1, 3).astype(np.float64)


    # uv, _ = cv2.projectPoints(
    #     xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    # )
    uv, _ = cv2.fisheye.projectPoints(
        xyz_cv, rvec, tvec, camera_matrix, dist_coeffs
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
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    # print(pixels)
    # Flip image horizontally
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]

    return pixels


def plot_trajs_and_points_on_image(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    list_trajs: list,
):
    """
    Plot trajectories and points on an image.
    """
    camera_height = 0.25
    camera_x_offset = 0.10

    for i, traj in enumerate(list_trajs):
        xy_coords = traj[:, :2]
        traj_pixels = get_pos_pixels(
            xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=False
        )
        
        
        points = traj_pixels.astype(int).reshape(-1, 1, 2)
        # print(f"points shape {points.shape}, traj_pixels shape {traj_pixels.shape}")
        # print(points[0])
        # print(points[:, :, ::-1][0])
        # points = points[:, :, ::-1]
        # Random color for each trajectory
        color = tuple(int(x) for x in np.random.choice(range(50, 255), size=3))

        # inverting x,y axis so origin in image is down-left corner
        points[:, :, 1] = VIZ_IMAGE_SIZE[1] - 1 - points[:, :, 1]

        # Draw trajectory
        cv2.polylines(img, [points], isClosed=False, color=color, thickness=3)

        # Draw start point (green) and goal point (red)
        # start = tuple(points[0, 0])
        # goal = tuple(points[-1, 0])
        # cv2.circle(img, start, 6, (0, 255, 0), -1)
        # cv2.circle(img, goal, 6, (0, 0, 255), -1)

    return img


if __name__ == "__main__":
    # Load or create test image
    try:
        read_image = cv2.imread("../debug_viz/obs_img.png")
        if read_image is None:
            raise FileNotFoundError
    except:
        print("ISSUE WHEN READING IMG..")
        read_image = np.ones((VIZ_IMAGE_SIZE[1], VIZ_IMAGE_SIZE[0], 3), dtype=np.uint8) * 255


    trajs = np.array([[
        [7.4099654e-01, 2.5383472e-02],
        [1.4056203e+00, 1.8866301e-01],
        [2.0843019e+00, 5.3543425e-01],
        [ 2.9190621e+00,  1.0020099e+00],                                                                                                   
        [ 3.6940556e+00,  1.7189999e+00],                                                                                                   
        [ 4.4188185e+00,  2.5908766e+00],                                                                                                   
        [ 5.0786438e+00,  3.5835190e+00],                                                                                                   
        [ 5.4698219e+00,  4.5496230e+00]
    ],
    [
        [6.0313344e-01, -6.4148903e-03],
        [1.3047167e+00,  4.6999931e-02],
        [2.0500667e+00,  2.4336529e-01],
        [ 2.7621977e+00,  6.1675119e-01],                                                                                                   
        [ 3.4182975e+00,  1.1521077e+00],                                                                                                   
        [ 4.0218000e+00,  1.8394799e+00],                                                                                                   
        [ 4.5111017e+00,  2.6454668e+00],                                                                                                   
        [ 4.9162483e+00,  3.5158238e+00]
    ],
    [
    
        [ 9.2684746e-01,  5.8219433e-02],                                                                                                   
        [ 1.8072345e+00,  2.9399157e-01],                                                                                                   
        [ 2.7151117e+00,  7.0648098e-01],                                                                                                   
        [ 3.6167743e+00,  1.2536325e+00],                                                                                                   
        [ 4.4826689e+00,  1.9135695e+00],                                                                                                   
        [ 5.2703495e+00,  2.6420469e+00],                                                                                                   
        [ 6.0599914e+00,  3.4693270e+00],                                                                                                   
        [ 6.6918368e+00,  4.3347516e+00]
    
    ],                                                                                                  
                                                                                                                                        
    [
    
        [ 6.6468990e-01,  2.4346828e-02],                                                                                                   
        [ 1.4727311e+00,  1.9263935e-01],                                                                                                   
        [ 2.2434640e+00,  5.8874416e-01],                                                                                                   
        [ 2.8736033e+00,  1.2336898e+00],                                                                                                   
        [ 3.4096651e+00,  2.0835962e+00],                                                                                                   
        [ 3.8411040e+00,  3.0542097e+00],                                                                                                   
        [ 4.2330580e+00,  4.1942873e+00],                                                                                                   
        [ 4.4533434e+00,  5.3529358e+00]
        
    ],

    [
        [ 7.9675722e-01,  9.6796036e-02],                                 
        [ 1.6490413e+00,  3.5488749e-01], 
        [ 2.4678710e+00,  7.6589203e-01], 
        [ 3.2462108e+00,  1.2792673e+00], 
        [ 3.9434361e+00,  1.9735060e+00], 
        [ 4.5293617e+00,  2.7676044e+00],                                 
        [ 5.0510907e+00,  3.5492477e+00], 
        [ 5.4978890e+00,  4.2636776e+00]
    ],                                
                                                                
    [
        [ 6.7781532e-01,  5.8173180e-02],                                 
        [ 1.3782953e+00,  2.2031736e-01], 
        [ 2.0777760e+00,  5.2858210e-01], 
        [ 2.8901405e+00,  1.0334845e+00], 
        [ 3.6263039e+00,  1.7050319e+00], 
        [ 4.2627263e+00,  2.5108190e+00],                                 
        [ 4.7641759e+00,  3.3184080e+00], 
        [ 5.1737089e+00,  4.2275085e+00]
    
    ],                                
                                                                    
    [
        [ 7.4134588e-01, -1.6517639e-02],                                 
        [ 1.5482219e+00,  3.2079697e-02], 
        [ 2.3174789e+00,  2.5250626e-01], 
        [ 3.0460446e+00,  6.6724634e-01], 
        [ 3.7531602e+00,  1.2710171e+00], 
        [ 4.3592687e+00,  1.9591932e+00],                                 
        [ 4.9461575e+00,  2.7672520e+00], 
        [ 5.4086499e+00,  3.5645542e+00]
    
    ],                                                                                                
    [
        
        [ 8.4416056e-01,  5.9240818e-02],                                 
        [ 1.6835381e+00,  1.6404963e-01], 
        [ 2.4458563e+00,  4.5645905e-01], 
        [ 3.1758025e+00,  1.0670323e+00], 
        [ 3.8571143e+00,  1.7949529e+00], 
        [ 4.3510361e+00,  2.5615482e+00], 
        [ 4.9055753e+00,  3.4006677e+00], 
        [ 5.5298200e+00,  4.2646112e+00]
        
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

    fig, ax = plt.subplots()
    img = plot_trajs_and_points_on_image(
        read_image,
        camera_matrix,
        dist_coeffs,
        trajs
    )

    ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
    ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))
    ax.imshow(img)
    plt.savefig('../debug_viz/img_test2.png')


    ros_img = bridge.cv2_to_imgmsg(img, encoding="bgr8")
    cv_img = bridge.imgmsg_to_cv2(ros_img, desired_encoding="bgr8")
    cv2.imwrite("../debug_viz/img_ros_test2.png", cv_img)
