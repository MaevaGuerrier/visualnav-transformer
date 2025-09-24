import numpy as np
from typing import Tuple
from viz_utils import get_pos_pixels
import torch
import torch.nn.functional as F
from threading import Event

# ROS
import ros_numpy
import rospy
from sensor_msgs.msg import Image

def get_traj_pixels_coords(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    list_trajs: list,
    viz_img_size: Tuple[int, int],
    resize_factor:bool=True):
    
    traj_pix_coords = []
   
    camera_height = 0.25
    camera_x_offset = 0.10

    for traj in list_trajs:
        # print("traj shape:", traj.shape)
        # print("traj:", traj)
        xy_coords = traj[:, :2]
        traj_pixels = get_pos_pixels(
            xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, viz_img_size
        )
        
        if resize_factor: # Traversability image is 224 x 224 and the original fisheye image is 640 x 480
            traj_pixels[:,0] *= .35
            traj_pixels[:,1] *= .46

        points = traj_pixels.astype(int).reshape(-1, 1, 2)

        # inverting x,y axis so origin in image is down-left corner
        if resize_factor:
            points[:, :, 1] = viz_img_size[1] * .46  - 1 - points[:, :, 1]
        else:
            points[:, :, 1] = viz_img_size[1] - 1 - points[:, :, 1]

        # Draw trajectory
        traj_pix_coords.append(points)

    return traj_pix_coords


def select_traj_best_traversability(
    trav_img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    list_trajs: list,
    viz_img_size: Tuple[int, int],
    resize_factor:bool=True
):
    """
    Select the trajectory with the best traversability score.
    trav_img: 2D numpy array with traversability values in range [0, 1] (0 = untraversable, 1 = fully traversable)
    """
    # TODO there might be a cleaner way to do this
    if trav_img is None:
        return None, None
    
    traj_pix_coords = get_traj_pixels_coords(
        camera_matrix, dist_coeffs, list_trajs, viz_img_size, resize_factor=resize_factor
    )

    best_traj = None
    best_score = -np.inf

    for traj, pix_coords in zip(list_trajs, traj_pix_coords):
        score = 0.0
        for point in pix_coords:
            x, y = point[0]
            score += trav_img[y, x] 
        rospy.logdebug(f"Trajectory score: {score}")
        if score > best_score:
            best_score = score
            best_traj = traj

    return best_traj, best_score



class traversabilityImageSubscriber:
    def __init__(self):
        self.trav_img = None
        self.overlay_traj_img = None
        self._trav_img_ready = Event() 

        rospy.Subscriber("/wild_visual_navigation_node/front/traversability", Image, self._callback_traversability_image, queue_size=10) 
        rospy.Subscriber("/wild_visual_navigation_visu_traversability_front/traversability_overlayed", Image, self._callback_traversability_overlay_image, queue_size=10)

        rospy.loginfo("Waiting for traversability image to be processed at least once...")
        self._trav_img_ready.wait()


    # 0 is untraversable and 1 is fully traversable. https://arxiv.org/pdf/2404.07110
    def _callback_traversability_image(self, trav_img_msg: Image):
        trav_img = ros_numpy.numpify(trav_img_msg)
        self.trav_img = trav_img if trav_img is not None else self.trav_img # In case we have messages dropping but only when master slave setup on robot not an issue
        is_in_range = np.all((trav_img >= 0) & (trav_img <= 1))
        assert is_in_range, "Traversability image values are out of range [0, 1]"
        self._trav_img_ready.set()
        rospy.logdebug(f"Received traversability image of shape: {trav_img.shape}")
        rospy.logdebug(f"Traversability image data type: {trav_img.dtype}\n")

    def _callback_traversability_overlay_image(self, trav_img_msg: Image):
        self.overlay_traj_img = ros_numpy.numpify(trav_img_msg)
        rospy.logdebug(f"Received traversability overlay image of shape: {self.overlay_traj_img.shape}")

    def get_trav_img(self):
        return self.trav_img
    
    def get_overlay_traj_img(self):
        return self.overlay_traj_img
    







################ DEUGING HERE ###################



# def get_traj_pixels_coords_torch(
#     camera_matrix: torch.Tensor,
#     dist_coeffs: torch.Tensor,   # currently unused (fisheye distortion not implemented)
#     list_trajs: torch.Tensor,    # (batch, horizon, dim) trajectories
#     viz_img_size: tuple,
#     resize_factor: bool = True,
#     camera_height: float = 0.25,
#     camera_x_offset: float = 0.10,
# ):
#     """
#     Differentiable version: projects trajectories into pixel coordinates using torch ops only.
#     - Assumes pinhole model (no fisheye distortion).
#     - Keeps gradients wrt list_trajs.
#     """

#     B, H, D = list_trajs.shape
#     device = list_trajs.device

#     # camera intrinsics
#     fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
#     cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

#     # (x, y, z) points in camera frame
#     xy = list_trajs[..., :2]  # (B, H, 2)
#     z = torch.full((B, H, 1), camera_height, device=device, dtype=list_trajs.dtype)
#     xyz = torch.cat([xy, z], dim=-1)

#     # apply camera offset
#     xyz[..., 0] += camera_x_offset

#     # reorder to (y, -z, x) for cv2 compatibility
#     X = xyz[..., 1]
#     Y = -xyz[..., 2]
#     Z = xyz[..., 0]

#     # perspective projection
#     u = fx * (X / Z) + cx
#     v = fy * (Y / Z) + cy

#     # flip horizontally
#     u = viz_img_size[0] - u

#     # apply resize factors (same as your numpy version)
#     if resize_factor:
#         u = u * 0.35
#         v = v * 0.46

#     # invert y-axis so origin is bottom-left
#     if resize_factor:
#         v = viz_img_size[1] * 0.46 - 1 - v
#     else:
#         v = viz_img_size[1] - 1 - v

#     # final pixel coords
#     pixel_coords = torch.stack([u, v], dim=-1)  # (B, H, 2)

#     # normalize to [-1, 1] for grid_sample
#     W, H_img = viz_img_size
#     pixel_coords_norm = pixel_coords.clone()
#     pixel_coords_norm[..., 0] = 2.0 * (pixel_coords[..., 0] / (W - 1)) - 1.0
#     pixel_coords_norm[..., 1] = 2.0 * (pixel_coords[..., 1] / (H_img - 1)) - 1.0

#     # reshape to (B, H, 1, 2) for grid_sample
#     return pixel_coords_norm.unsqueeze(2)

# from typing import Tuple, List, Union
# ArrayLike = Union[np.ndarray, torch.Tensor]

# def get_traj_pixels_coords_TESTING(
#     camera_matrix: ArrayLike,
#     dist_coeffs: ArrayLike,
#     list_trajs: List[ArrayLike],
#     viz_img_size: Tuple[int, int],
#     resize_factor: bool = True
# ):
#     traj_pix_coords = []

#     camera_height = 0.25
#     camera_x_offset = 0.10

#     # Detect if inputs are tensors
#     return_tensors = any(isinstance(traj, torch.Tensor) for traj in list_trajs)

#     for traj in list_trajs:
#         if isinstance(traj, torch.Tensor):
#             traj_np = traj.detach().cpu().numpy()
#         else:
#             traj_np = traj

#         # print("traj shape:", traj_np.shape)
#         # print("traj:", traj_np)

#         xy_coords = traj_np[:, :2]

#         traj_pixels = get_pos_pixels(
#             xy_coords, camera_height, camera_x_offset,
#             camera_matrix, dist_coeffs, viz_img_size
#         )

#         if resize_factor:
#             traj_pixels[:, 0] *= 0.35
#             traj_pixels[:, 1] *= 0.46

#         points = traj_pixels.astype(int).reshape(-1, 1, 2)

#         # Invert y-axis so origin is bottom-left
#         if resize_factor:
#             points[:, :, 1] = viz_img_size[1] * 0.46 - 1 - points[:, :, 1]
#         else:
#             points[:, :, 1] = viz_img_size[1] - 1 - points[:, :, 1]

#         # Convert back to tensor if needed
#         if return_tensors:
#             points = torch.from_numpy(points)

#         traj_pix_coords.append(points)

#     if return_tensors:
#         return torch.stack(traj_pix_coords, dim=0)
#     else:
#         return traj_pix_coords


# def compute_traversability_guidance(naction, traversability_map, camera_matrix, 
#                                   dist_coeffs, viz_img_size):
#     """
#     Compute traversability guidance for trajectory batch
#     """
#     # Convert trajectories to pixel coordinates
#     batch_size = naction.shape[0]
#     pixel_coords = get_traj_pixels_coords_torch(
#         camera_matrix, dist_coeffs, naction, viz_img_size
#     )
    
#     if isinstance(traversability_map, np.ndarray):
#         traversability_map = torch.tensor(traversability_map, dtype=torch.float32)
#     if traversability_map.dim() == 2:  # (H, W)
#         traversability_map = traversability_map.unsqueeze(0).unsqueeze(0)

#     traversability_map = traversability_map.repeat(batch_size, 1, 1, 1)
#     print("traversability_map shape:", traversability_map.shape)
#     print("pixel_coords shape:", pixel_coords.shape)

#     # Sample traversability values at trajectory waypoints
#     # traversability_map should be (1, 1, H, W) for grid_sample
#     print("coords grad:", pixel_coords.requires_grad, pixel_coords.grad_fn)
#     print("traversability_map grad:", traversability_map.requires_grad, traversability_map.grad_fn)
#     traversability_values = F.grid_sample(
#         traversability_map,
#         pixel_coords.float(),  # (batch, naction, 1, 2)
#         mode='bilinear',
#         padding_mode='border',
#         align_corners=False
#     ).squeeze(-1).squeeze(0)  # (batch, naction)
#     print("traversability_values shape:", traversability_values.shape)
#     print("traversability_values grad:", traversability_values.requires_grad, traversability_values.grad_fn)
#     # Compute loss - encourage high traversability
#     # Add small epsilon to prevent log(0)
#     traversability_loss = -torch.log(traversability_values.clamp(min=1e-6)).mean()
    
#     return traversability_loss



def compute_traversability_score_batch(naction_batch, traversability_map, 
                                     camera_matrix, dist_coeffs, viz_img_size):
    """
    Compute traversability scores for a batch of trajectories
    Uses your existing non-differentiable projection function
    """
    batch_size = naction_batch.shape[0]
    scores = []
    
    # Convert tensors to numpy for your existing function
    camera_matrix_np = camera_matrix if isinstance(camera_matrix, np.ndarray) else camera_matrix.cpu().numpy()
    dist_coeffs_np = dist_coeffs if isinstance(dist_coeffs, np.ndarray) else dist_coeffs.cpu().numpy()
    
    # Get traversability map as numpy
    if isinstance(traversability_map, torch.Tensor):
        if traversability_map.dim() == 4:  # (batch, channel, H, W)
            trav_map_np = traversability_map[0, 0].cpu().numpy()
        elif traversability_map.dim() == 3:  # (channel, H, W) or (batch, H, W)
            trav_map_np = traversability_map[0].cpu().numpy()
        else:  # (H, W)
            trav_map_np = traversability_map.cpu().numpy()
    else:
        trav_map_np = traversability_map
    
    for i in range(batch_size):
        # Convert single trajectory to numpy
        traj = naction_batch[i].detach().cpu().numpy()
        
        # Use your existing function to get pixel coordinates
        try:
            pixel_coords_list = get_traj_pixels_coords(
                camera_matrix_np, dist_coeffs_np, [traj], viz_img_size, resize_factor=True
            )
            pixel_coords = pixel_coords_list[0].reshape(-1, 2)  # (naction_steps, 2)
            
            # Clamp coordinates to image bounds
            pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, trav_map_np.shape[1] - 1)
            pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, trav_map_np.shape[0] - 1)
            
            # Sample traversability values at pixel locations
            pixel_coords_int = pixel_coords.astype(int)
            trav_values = trav_map_np[pixel_coords_int[:, 1], pixel_coords_int[:, 0]]
            
            # Compute mean traversability score
            score = np.mean(trav_values)
            
        except Exception as e:
            print(f"Error computing traversability for batch {i}: {e}")
            score = 0.0  # Fallback to poor traversability
        
        scores.append(score)
    
    return torch.tensor(scores, device=naction_batch.device, dtype=torch.float32)


def compute_traversability_guidance_finite_diff(naction, traversability_map, camera_matrix, 
                                              dist_coeffs, viz_img_size, epsilon=0.01):
    """
    Use finite differences to approximate gradients for traversability guidance
    """
    _, naction_steps, coord_dim = naction.shape
    device = naction.device
    
    # print(f"Computing finite difference gradients with epsilon={epsilon}")
    
    # Current traversability scores for all trajectories in batch
    current_scores = compute_traversability_score_batch(
        naction, traversability_map, camera_matrix, dist_coeffs, viz_img_size
    )
    
    print(f"Current traversability scores: mean={current_scores.mean().item():.3f}, "
          f"min={current_scores.min().item():.3f}, max={current_scores.max().item():.3f}")
    
    # Initialize gradient tensor
    gradients = torch.zeros_like(naction, device=device)
    
    # Compute partial derivatives using finite differences
    for i in range(naction_steps):  # For each waypoint
        for j in range(coord_dim):  # For x and y coordinates
            # Create perturbed version (increase coordinate by epsilon)
            naction_plus = naction.clone()
            naction_plus[:, i, j] += epsilon
            
            # Compute traversability scores with perturbation
            scores_plus = compute_traversability_score_batch(
                naction_plus, traversability_map, camera_matrix, dist_coeffs, viz_img_size
            )
            
            # Finite difference gradient: (f(x+h) - f(x)) / h
            gradients[:, i, j] = (scores_plus - current_scores) / epsilon
    
    # Compute loss (negative mean score because we want to maximize traversability)
    loss = -current_scores.mean()
    
    print(f"Gradient norm: {gradients.norm().item():.6f}")
    print(f"Traversability loss: {loss.item():.6f}")

    return loss, gradients


def sample_with_traversability_guidance(model, noisy_action, noise_scheduler, obs_cond, 
                                       traversability_map, camera_matrix, dist_coeffs, 
                                       viz_img_size, guidance_scale=0.5, 
                                       finite_diff_epsilon=0.01):
    """
    Complete sampling function with traversability guidance using finite differences
    """
    # Initialize noise (adjust dimensions as needed)
    # naction = torch.randn(batch_size, naction_steps, 2, device=device)
    naction = noisy_action
    
    for i, k in enumerate(noise_scheduler.timesteps):
        print(f"\nTimestep {i+1}/{len(noise_scheduler.timesteps)}: {k}")
        
        # Don't need requires_grad for finite differences
        # naction.requires_grad_(False)
        
        # predict noise
        noise_pred = model(
            'noise_pred_net',
            sample=naction,
            timestep=k,
            global_cond=obs_cond
        )
        
        # Apply traversability guidance
        if guidance_scale > 0:
            try:
                # Compute guidance using finite differences
                trav_loss, traj_grad = compute_traversability_guidance_finite_diff(
                    naction, traversability_map, camera_matrix, dist_coeffs, 
                    viz_img_size, epsilon=finite_diff_epsilon
                )
                
                # Time-dependent guidance scaling (stronger early in sampling)
                timestep_ratio = i / len(noise_scheduler.timesteps)
                guidance_weight = guidance_scale * (1 - timestep_ratio)
                # guidance_weight = guidance_scale * (1 - timestep_ratio**0.5)
                # guidance_weight = guidance_scale * (0.5 * (1 + torch.cos(torch.tensor(timestep_ratio * 3.14159))))
                # guidance_weight = guidance_scale

                
                print(f"Applied guidance: loss={trav_loss.item():.4f}, "
                      f"grad_norm={traj_grad.norm().item():.6f}, weight={guidance_weight:.3f}")
                
                # Apply guidance (ADD gradient because we want to maximize traversability)
                # The gradient points in the direction of increasing traversability
                noise_pred = noise_pred + guidance_weight * traj_grad
                
            except Exception as e:
                print(f"Guidance failed at timestep {i}: {e}")
                print("Continuing without guidance for this step")
        
        # inverse diffusion step (remove noise)
        naction = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=naction
        ).prev_sample
        
        # Optional: Evaluate current trajectory quality
        if i % 10 == 0 or i == len(noise_scheduler.timesteps) - 1:
            with torch.no_grad():
                current_scores = compute_traversability_score_batch(
                    naction, traversability_map, camera_matrix, dist_coeffs, viz_img_size
                )
                print(f"Current trajectory traversability: {current_scores.mean().item():.3f}")
                print(f"Scores range: min={current_scores.min().item():.3f}, max={current_scores.max().item():.3f}")
                # print(f"Highest scoring trajectory index: {torch.argmax(current_scores).item()}")

    return naction



camera_matrix_orig = np.array([
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

VIZ_IMAGE_SIZE_FISHEYE = (640, 480) 


def get_traversability_grad_cost(
    trajs: torch.Tensor,       # [B, T, 2], pixel coords in 224x224 image space
    trav_map: torch.Tensor     # [H, W] numpy or torch with values [0,1]
):
    """
    Differentiable traversability guidance.
    trajs: trajectories in pixel coordinates of traversability map (e.g., 224x224).
    trav_map: traversability image (0=not traversable, 1=traversable).
    """

    device = trajs.device
    B, T, _ = trajs.shape

    trajs = trajs.detach().cpu().numpy()
    traj_pixel_coords = get_traj_pixels_coords(
        camera_matrix_orig, dist_coeffs, trajs, VIZ_IMAGE_SIZE_FISHEYE, resize_factor=True
    )
    traj_pixel_coords = np.array(traj_pixel_coords)  # [B,T,1,2]
    traj_pixel_coords = torch.tensor(traj_pixel_coords, dtype=torch.float32, device=device)  # [B,T,1,2]

    # Convert traversability map (numpy -> torch if needed)
    if isinstance(trav_map, np.ndarray):
        trav_map = torch.tensor(trav_map, dtype=torch.float32, device=device)

    if trav_map.dim() == 2:
        trav_map = trav_map.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    H, W = trav_map.shape[-2:]

    # Ensure trajs require gradients
    trajs_in = traj_pixel_coords.detach().requires_grad_(True).to(device).squeeze()
    trajs_in.retain_grad()
    # print("TRAJ TYPE:", type(trajs_in))

    # Normalize trajs into [-1,1] for grid_sample
    grid = trajs_in.clone()
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  # x
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  # y
    grid = grid.unsqueeze(1)  # [B,1,T,2]

    # Sample traversability values along trajectories
    # print all grad info
    # print(f"trav map is of type: {type(trav_map)} and shape: {trav_map.shape}")
    # print("trajs_in grad:", trajs_in.requires_grad, trajs_in.grad_fn)
    # print("trav_map grad:", trav_map.requires_grad, trav_map.grad_fn)
    # print("grid grad:", grid.requires_grad, grid.grad_fn)
    sampled = F.grid_sample(
        trav_map.expand(B, -1, -1, -1),  # [B,1,H,W]
        grid,                            # [B,1,T,2]
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    ).squeeze(1).squeeze(1)  # -> [B,T]
    sampled = sampled.to(torch.float32)  

    # print("AFTER GRID SAMPLE")

    # Define loss = negative traversability (maximize traversability)
    loss = -sampled.mean()
    # Backprop to get gradients wrt trajs
    # print("BEFORE BACKWARD")
    # print(f"GRAD LOSS:", loss.requires_grad, loss.grad_fn)
    loss.backward()
    # print(f"GRAD LOSS:", loss.requires_grad, loss.grad_fn)

    traj_grad = trajs_in.grad

    return loss.item(), traj_grad


def get_traversability_grad_cost_finite_diff(trajs, trav_map, camera_matrix_orig, 
                                                    dist_coeffs, viz_img_size, epsilon=0.01):
    """
    Fixed version using finite differences that works with your existing projection
    
    Args:
        trajs: [B, T, 2] trajectory tensor in world coordinates 
        trav_map: [H, W] traversability map (numpy or torch)
        epsilon: finite difference step size
        
    Returns:
        loss: scalar loss value
        traj_grad: [B, T, 2] gradients w.r.t. input trajs
    """
    device = trajs.device
    B, T, _ = trajs.shape
    
    # Convert trav_map to numpy for your existing projection function
    if isinstance(trav_map, torch.Tensor):
        trav_map_np = trav_map.cpu().numpy()
    else:
        trav_map_np = trav_map
        
    # Convert camera params to numpy
    camera_matrix_np = camera_matrix_orig if isinstance(camera_matrix_orig, np.ndarray) else camera_matrix_orig.cpu().numpy()
    dist_coeffs_np = dist_coeffs if isinstance(dist_coeffs, np.ndarray) else dist_coeffs.cpu().numpy()
    
    def evaluate_traversability_batch(traj_batch):
        """Helper to evaluate traversability for a batch of trajectories"""
        traj_batch_np = traj_batch.detach().cpu().numpy()
        scores = []
        
        for i in range(B):
            try:
                # Use your existing projection function
                pixel_coords_list = get_traj_pixels_coords(
                    camera_matrix_np, dist_coeffs_np, [traj_batch_np[i]], viz_img_size, resize_factor=True
                )
                pixel_coords = pixel_coords_list[0].reshape(-1, 2)  # [T, 2]
                
                # Clamp to image bounds
                pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, trav_map_np.shape[1] - 1)
                pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, trav_map_np.shape[0] - 1)
                
                # Sample traversability values
                pixel_coords_int = pixel_coords.astype(int)
                trav_values = trav_map_np[pixel_coords_int[:, 1], pixel_coords_int[:, 0]]
                
                # Compute mean traversability for this trajectory
                score = np.mean(trav_values)
                
            except Exception as e:
                print(f"Error evaluating traversability for batch {i}: {e}")
                score = 0.0  # Poor traversability as fallback
                
            scores.append(score)
        
        return torch.tensor(scores, device=device, dtype=torch.float32)
    
    # Evaluate current traversability
    current_scores = evaluate_traversability_batch(trajs)
    
    # Initialize gradient tensor
    traj_grad = torch.zeros_like(trajs)
    
    # Compute finite difference gradients
    for t in range(T):  # For each waypoint
        for dim in range(2):  # For x and y coordinates
            # Create perturbed trajectory
            trajs_plus = trajs.clone()
            trajs_plus[:, t, dim] += epsilon
            
            # Evaluate traversability with perturbation
            scores_plus = evaluate_traversability_batch(trajs_plus)
            
            # Finite difference gradient: (f(x+h) - f(x)) / h
            traj_grad[:, t, dim] = (scores_plus - current_scores) / epsilon
    
    # Loss is negative mean traversability (we want to maximize traversability)
    loss = -current_scores.mean()
    
    print(f"Traversability scores: mean={current_scores.mean().item():.3f}, "
          f"min={current_scores.min().item():.3f}, max={current_scores.max().item():.3f}")
    print(f"Gradient norm: {traj_grad.norm().item():.6f}")
    
    return loss.item(), traj_grad


def get_traversability_grad_cost_simple(trajs, trav_map, camera_matrix_orig, 
                                       dist_coeffs, viz_img_size, epsilon=0.02):
    """
    Simplified version - even more robust
    """
    device = trajs.device
    B, T, _ = trajs.shape
    
    # Convert everything to numpy for your existing functions
    trajs_np = trajs.detach().cpu().numpy()
    trav_map_np = trav_map.cpu().numpy() if isinstance(trav_map, torch.Tensor) else trav_map
    camera_matrix_np = camera_matrix_orig if isinstance(camera_matrix_orig, np.ndarray) else camera_matrix_orig.cpu().numpy()
    dist_coeffs_np = dist_coeffs if isinstance(dist_coeffs, np.ndarray) else dist_coeffs.cpu().numpy()
    
    def single_traj_score(traj_single):
        """Evaluate single trajectory traversability"""
        try:
            pixel_coords_list = get_traj_pixels_coords(
                camera_matrix_np, dist_coeffs_np, [traj_single], viz_img_size, resize_factor=True
            )
            pixel_coords = pixel_coords_list[0].reshape(-1, 2)
            
            # Clamp and sample
            pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, trav_map_np.shape[1] - 1)
            pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, trav_map_np.shape[0] - 1)
            
            pixel_coords_int = pixel_coords.astype(int)
            trav_values = trav_map_np[pixel_coords_int[:, 1], pixel_coords_int[:, 0]]
            
            return np.mean(trav_values)
        except:
            return 0.0
    
    # Compute current scores and gradients
    current_scores = []
    gradients = np.zeros_like(trajs_np)
    
    for b in range(B):
        # Current score for this trajectory
        current_score = single_traj_score(trajs_np[b])
        current_scores.append(current_score)
        
        # Compute gradients for each waypoint
        for t in range(T):
            for dim in range(2):
                # Perturb this coordinate
                trajs_perturbed = trajs_np[b].copy()
                trajs_perturbed[t, dim] += epsilon
                
                # Evaluate perturbed score
                perturbed_score = single_traj_score(trajs_perturbed)
                
                # Finite difference gradient
                gradients[b, t, dim] = (perturbed_score - current_score) / epsilon
    
    # Convert results back to torch tensors
    current_scores = torch.tensor(current_scores, device=device, dtype=torch.float32)
    traj_grad = torch.tensor(gradients, device=device, dtype=torch.float32)
    
    # Loss (negative because we want to maximize traversability)
    loss = -current_scores.mean()
    
    print(f"Simple traversability - mean: {current_scores.mean().item():.3f}, grad_norm: {traj_grad.norm().item():.6f}")
    
    return loss.item(), traj_grad


# Updated main sampling loop
def fixed_sampling_loop(model, obs_images, fake_goal, mask, model_params, device, num_diffusion_iters,
                        args, noise_scheduler, trav_img_subscriber):
    """
    Fixed version of your sampling loop
    """
    with torch.no_grad():
        # encoder vision features
        obs_cond = model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
        
        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
        
        # initialize action from Gaussian noise
        noisy_action = torch.randn(
            (args.num_samples, model_params["len_traj_pred"], 2), device=device)
        naction = noisy_action

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

    start_time = time.time()
    
    for i, k in enumerate(noise_scheduler.timesteps):
        print(f"\nDiffusion step {i+1}/{len(noise_scheduler.timesteps)}")
        
        # --------------------------------------------------------
        # 1. Diffusion prediction step
        # --------------------------------------------------------
        with torch.no_grad():
            noise_pred = model(
                'noise_pred_net',
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # --------------------------------------------------------
        # 2. Guidance step - FIXED VERSION
        # --------------------------------------------------------
        if args.enable_trav:
            print("Applying traversability guidance...")
            
            # Get traversability map
            trav_map = trav_img_subscriber.get_trav_img()  # Keep as numpy
            
            # Compute gradients using finite differences
            trav_loss, traj_grad = get_traversability_grad_cost_simple(
                naction,  # This is the correct variable to get gradients for
                trav_map,
                camera_matrix_orig,
                dist_coeffs,
                VIZ_IMAGE_SIZE_FISHEYE,
                epsilon=0.02  # Tune this if needed
            )

            # Apply guidance with time-dependent scaling
            timestep_ratio = i / len(noise_scheduler.timesteps)
            grad_scale = 1.0 * (1 - timestep_ratio)  # Stronger guidance early
            
            print(f"Applying guidance with scale: {grad_scale:.3f}")
            
            # FIXED: Add gradients (not subtract) to move toward better traversability
            # The gradient points in direction of increasing traversability
            naction = naction + grad_scale * traj_grad
            
            # Optional: Clamp trajectories to reasonable bounds
            # naction = torch.clamp(naction, min=-5.0, max=5.0)  # Adjust bounds as needed
    
    return naction


# Alternative: Even simpler debugging version
def debug_traversability_effect(trav_img_subscriber, device):
    """
    Simple test to verify traversability guidance is working
    """
    # Create test trajectory
    test_traj = torch.randn(1, 8, 2, device=device) * 2  # Single trajectory
    
    print("=== DEBUGGING TRAVERSABILITY EFFECT ===")
    print(f"Original trajectory: {test_traj[0, :3, :]}...")  # First 3 waypoints
    
    # Evaluate original traversability
    trav_map = trav_img_subscriber.get_trav_img()
    original_loss, original_grad = get_traversability_grad_cost_simple(
        test_traj, trav_map, camera_matrix_orig, dist_coeffs, VIZ_IMAGE_SIZE_FISHEYE
    )
    
    print(f"Original traversability loss: {original_loss:.4f}")
    print(f"Gradient magnitude: {original_grad.norm().item():.6f}")
    
    # Apply guidance
    guidance_scale = 1.0
    improved_traj = test_traj + guidance_scale * original_grad
    
    print(f"Improved trajectory: {improved_traj[0, :3, :]}...")
    
    # Evaluate improved traversability
    improved_loss, _ = get_traversability_grad_cost_simple(
        improved_traj, trav_map, camera_matrix_orig, dist_coeffs, VIZ_IMAGE_SIZE_FISHEYE
    )
    
    print(f"Improved traversability loss: {improved_loss:.4f}")
    print(f"Improvement: {original_loss - improved_loss:.4f}")
    
    if improved_loss < original_loss:
        print("✅ Traversability guidance is working!")
    else:
        print("❌ Traversability guidance may not be working properly")
    
    return test_traj, improved_traj



############################# DEBUG WORKING USING FINITE DIFF DEBUGING PIPELINE #############################
import numpy as np
import torch
import matplotlib.pyplot as plt

def comprehensive_debug_traversability(naction, trav_map, camera_matrix_orig, 
                                     dist_coeffs, viz_img_size, epsilon=0.02):
    """
    Comprehensive debugging of traversability guidance
    """
    print("="*60)
    print("COMPREHENSIVE TRAVERSABILITY DEBUG")
    print("="*60)
    
    # 1. Check input validity
    print(f"1. INPUT VALIDATION:")
    print(f"   naction shape: {naction.shape}, dtype: {naction.dtype}, device: {naction.device}")
    print(f"   naction range: [{naction.min().item():.3f}, {naction.max().item():.3f}]")
    print(f"   trav_map shape: {trav_map.shape if hasattr(trav_map, 'shape') else 'numpy'}")
    print(f"   trav_map range: [{np.min(trav_map):.3f}, {np.max(trav_map):.3f}]")
    
    # 2. Test projection function
    print(f"\n2. PROJECTION FUNCTION TEST:")
    test_traj = naction[0].detach().cpu().numpy()  # First trajectory
    
    try:
        camera_matrix_np = camera_matrix_orig if isinstance(camera_matrix_orig, np.ndarray) else camera_matrix_orig.cpu().numpy()
        dist_coeffs_np = dist_coeffs if isinstance(dist_coeffs, np.ndarray) else dist_coeffs.cpu().numpy()
        
        pixel_coords_list = get_traj_pixels_coords(
            camera_matrix_np, dist_coeffs_np, [test_traj], viz_img_size, resize_factor=True
        )
        pixel_coords = pixel_coords_list[0].reshape(-1, 2)
        
        print(f"   ✅ Projection successful")
        print(f"   Pixel coords shape: {pixel_coords.shape}")
        print(f"   Pixel coords range: x=[{pixel_coords[:, 0].min():.1f}, {pixel_coords[:, 0].max():.1f}], "
              f"y=[{pixel_coords[:, 1].min():.1f}, {pixel_coords[:, 1].max():.1f}]")
        print(f"   Image bounds: [0, {trav_map.shape[1]-1}] x [0, {trav_map.shape[0]-1}]")
        
        # Check if trajectory is within image bounds
        in_bounds_x = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < trav_map.shape[1])
        in_bounds_y = (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < trav_map.shape[0])
        in_bounds = in_bounds_x & in_bounds_y
        
        print(f"   Waypoints in bounds: {np.sum(in_bounds)}/{len(in_bounds)} ({100*np.mean(in_bounds):.1f}%)")
        
        if np.sum(in_bounds) == 0:
            print(f"   ⚠️  WARNING: No waypoints are within image bounds!")
            return None, None
        
    except Exception as e:
        print(f"   ❌ Projection failed: {e}")
        return None, None
    
    # 3. Test traversability sampling
    print(f"\n3. TRAVERSABILITY SAMPLING TEST:")
    try:
        # Clamp to bounds
        pixel_coords_clamped = pixel_coords.copy()
        pixel_coords_clamped[:, 0] = np.clip(pixel_coords_clamped[:, 0], 0, trav_map.shape[1] - 1)
        pixel_coords_clamped[:, 1] = np.clip(pixel_coords_clamped[:, 1], 0, trav_map.shape[0] - 1)
        
        pixel_coords_int = pixel_coords_clamped.astype(int)
        trav_values = trav_map[pixel_coords_int[:, 1], pixel_coords_int[:, 0]]
        
        print(f"   ✅ Sampling successful")
        print(f"   Trav values shape: {trav_values.shape}")
        print(f"   Trav values range: [{trav_values.min():.3f}, {trav_values.max():.3f}]")
        print(f"   Trav values mean: {trav_values.mean():.3f}")
        print(f"   Non-zero values: {np.sum(trav_values > 0)}/{len(trav_values)}")
        
        if np.all(trav_values == 0):
            print(f"   ⚠️  WARNING: All traversability values are 0!")
        elif np.all(trav_values == 1):
            print(f"   ⚠️  WARNING: All traversability values are 1!")
        
    except Exception as e:
        print(f"   ❌ Traversability sampling failed: {e}")
        return None, None
    
    # 4. Test finite difference gradients
    print(f"\n4. FINITE DIFFERENCE GRADIENT TEST:")
    
    def evaluate_single_traj(traj_np):
        try:
            pixel_coords_list = get_traj_pixels_coords(
                camera_matrix_np, dist_coeffs_np, [traj_np], viz_img_size, resize_factor=True
            )
            pixel_coords = pixel_coords_list[0].reshape(-1, 2)
            pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, trav_map.shape[1] - 1)
            pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, trav_map.shape[0] - 1)
            pixel_coords_int = pixel_coords.astype(int)
            trav_values = trav_map[pixel_coords_int[:, 1], pixel_coords_int[:, 0]]
            return np.mean(trav_values)
        except:
            return 0.0
    
    original_score = evaluate_single_traj(test_traj)
    print(f"   Original score: {original_score:.4f}")
    
    # Test gradients for first waypoint, both dimensions
    test_gradients = []
    for dim in range(2):
        test_traj_perturbed = test_traj.copy()
        test_traj_perturbed[0, dim] += epsilon  # Perturb first waypoint
        
        perturbed_score = evaluate_single_traj(test_traj_perturbed)
        gradient = (perturbed_score - original_score) / epsilon
        test_gradients.append(gradient)
        
        print(f"   Dim {dim}: original={original_score:.4f}, perturbed={perturbed_score:.4f}, grad={gradient:.6f}")
    
    gradient_magnitude = np.sqrt(sum(g**2 for g in test_gradients))
    print(f"   Gradient magnitude: {gradient_magnitude:.6f}")
    
    if gradient_magnitude < 1e-8:
        print(f"   ⚠️  WARNING: Gradient magnitude very small - may not provide useful guidance")
    elif gradient_magnitude > 10:
        print(f"   ⚠️  WARNING: Gradient magnitude very large - may cause instability")
    
    # 5. Test multiple random perturbations
    print(f"\n5. ROBUSTNESS TEST (10 random perturbations):")
    robustness_scores = []
    for _ in range(10):
        # Create random perturbation
        random_traj = test_traj + np.random.normal(0, 0.1, test_traj.shape)
        score = evaluate_single_traj(random_traj)
        robustness_scores.append(score)
    
    robustness_scores = np.array(robustness_scores)
    print(f"   Score variance: {robustness_scores.var():.6f}")
    print(f"   Score range: [{robustness_scores.min():.4f}, {robustness_scores.max():.4f}]")
    
    if robustness_scores.var() < 1e-6:
        print(f"   ⚠️  WARNING: Very low score variance - traversability map may be uniform")
    
    # 6. Visualize trajectory on traversability map
    print(f"\n6. TRAJECTORY VISUALIZATION:")
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot traversability map
        ax1.imshow(trav_map, cmap='viridis', origin='upper')
        ax1.set_title('Traversability Map')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        
        # Overlay trajectory
        if np.sum(in_bounds) > 0:
            valid_coords = pixel_coords_clamped[in_bounds]
            ax1.plot(valid_coords[:, 0], valid_coords[:, 1], 'r-', linewidth=2, alpha=0.8, label='Trajectory')
            ax1.scatter(valid_coords[:, 0], valid_coords[:, 1], c='red', s=20, alpha=0.8)
        
        ax1.legend()
        
        # Plot traversability values along trajectory
        ax2.plot(trav_values, 'b-', marker='o', linewidth=2, markersize=4)
        ax2.set_title('Traversability Along Trajectory')
        ax2.set_xlabel('Waypoint Index')
        ax2.set_ylabel('Traversability Value')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('traversability_debug.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Visualization saved to 'traversability_debug.png'")
        
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {e}")
    
    # 7. Final assessment
    print(f"\n7. FINAL ASSESSMENT:")
    
    issues_found = []
    
    if np.sum(in_bounds) < len(in_bounds) * 0.8:
        issues_found.append("Many waypoints outside image bounds")
    
    if np.all(trav_values == 0) or np.all(trav_values == 1):
        issues_found.append("Uniform traversability values")
    
    if gradient_magnitude < 1e-6:
        issues_found.append("Gradients too small")
    elif gradient_magnitude > 50:
        issues_found.append("Gradients too large")
    
    if robustness_scores.var() < 1e-6:
        issues_found.append("Low traversability variance")
    
    if len(issues_found) == 0:
        print(f"   ✅ All checks passed - traversability guidance should work!")
        status = "GOOD"
    else:
        print(f"   ⚠️  Issues found:")
        for issue in issues_found:
            print(f"      - {issue}")
        status = "ISSUES"
    
    # Return diagnostic info
    return {
        'status': status,
        'original_score': original_score,
        'gradient_magnitude': gradient_magnitude,
        'in_bounds_ratio': np.mean(in_bounds),
        'trav_value_range': (trav_values.min(), trav_values.max()),
        'score_variance': robustness_scores.var(),
        'issues': issues_found
    }, test_gradients


def improved_traversability_guidance(naction, trav_map, camera_matrix_orig, 
                                   dist_coeffs, viz_img_size, epsilon=0.02):
    """
    Improved version with better error handling and diagnostics
    """
    device = naction.device
    B, T, _ = naction.shape
    
    # Diagnostic info
    diagnostics, _ = comprehensive_debug_traversability(
        naction, trav_map, camera_matrix_orig, dist_coeffs, viz_img_size, epsilon
    )
    
    if diagnostics is None or diagnostics['status'] != 'GOOD':
        print("❌ Traversability guidance may not work reliably")
        return 0.0, torch.zeros_like(naction)
    
    # Convert inputs
    trav_map_np = trav_map.cpu().numpy() if isinstance(trav_map, torch.Tensor) else trav_map
    camera_matrix_np = camera_matrix_orig if isinstance(camera_matrix_orig, np.ndarray) else camera_matrix_orig.cpu().numpy()
    dist_coeffs_np = dist_coeffs if isinstance(dist_coeffs, np.ndarray) else dist_coeffs.cpu().numpy()
    
    def robust_single_traj_score(traj_single):
        """Robust single trajectory evaluation with fallbacks"""
        try:
            pixel_coords_list = get_traj_pixels_coords(
                camera_matrix_np, dist_coeffs_np, [traj_single], viz_img_size, resize_factor=True
            )
            pixel_coords = pixel_coords_list[0].reshape(-1, 2)
            
            # Check bounds
            in_bounds_x = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < trav_map_np.shape[1])
            in_bounds_y = (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < trav_map_np.shape[0])
            in_bounds = in_bounds_x & in_bounds_y
            
            if np.sum(in_bounds) == 0:
                return 0.0  # All points out of bounds
            
            # Clamp to bounds
            pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, trav_map_np.shape[1] - 1)
            pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, trav_map_np.shape[0] - 1)
            
            pixel_coords_int = pixel_coords.astype(int)
            trav_values = trav_map_np[pixel_coords_int[:, 1], pixel_coords_int[:, 0]]
            
            # Weight by in-bounds ratio
            score = np.mean(trav_values) * (np.sum(in_bounds) / len(in_bounds))
            
            return float(score)
            
        except Exception as e:
            print(f"Error in trajectory evaluation: {e}")
            return 0.0
    
    # Compute scores and gradients
    trajs_np = naction.detach().cpu().numpy()
    current_scores = []
    gradients = np.zeros_like(trajs_np)
    
    for b in range(B):
        current_score = robust_single_traj_score(trajs_np[b])
        current_scores.append(current_score)
        
        # Compute gradients with adaptive epsilon
        adaptive_epsilon = epsilon
        
        for t in range(T):
            for dim in range(2):
                # Try different epsilon values if gradient is too small
                for eps_scale in [1.0, 2.0, 0.5]:
                    test_epsilon = adaptive_epsilon * eps_scale
                    
                    trajs_perturbed = trajs_np[b].copy()
                    trajs_perturbed[t, dim] += test_epsilon
                    
                    perturbed_score = robust_single_traj_score(trajs_perturbed)
                    gradient = (perturbed_score - current_score) / test_epsilon
                    
                    if abs(gradient) > 1e-8:  # Found meaningful gradient
                        gradients[b, t, dim] = gradient
                        break
                else:
                    gradients[b, t, dim] = 0.0  # No meaningful gradient found
    
    # Convert back to tensors
    current_scores = torch.tensor(current_scores, device=device, dtype=torch.float32)
    traj_grad = torch.tensor(gradients, device=device, dtype=torch.float32)
    
    loss = -current_scores.mean()
    
    print(f"Robust guidance - scores: mean={current_scores.mean().item():.4f}, "
          f"grad_norm={traj_grad.norm().item():.6f}")
    
    return loss.item(), traj_grad


# Usage in your main loop
def debug_and_apply_guidance(naction, trav_map, camera_matrix_orig, dist_coeffs, viz_img_size):
    """
    Debug first, then apply guidance
    """
    # Run comprehensive debug every N steps or when issues occur
    diagnostics, _ = comprehensive_debug_traversability(
        naction, trav_map, camera_matrix_orig, dist_coeffs, viz_img_size
    )
    
    if diagnostics and diagnostics['status'] == 'GOOD':
        # Use improved guidance
        return improved_traversability_guidance(
            naction, trav_map, camera_matrix_orig, dist_coeffs, viz_img_size
        )
    else:
        print("❌ Skipping guidance due to issues")
        return 0.0, torch.zeros_like(naction)
    


    ######################## WORKING DEBUG LOOP CODE #########################










############# TYRING ENHANCE TRAV ###############################################







def adaptive_finite_difference_gradients(naction, trav_map, camera_matrix_orig, 
                                        dist_coeffs, viz_img_size, 
                                        base_epsilon=0.02, epsilon_scales=[1.0, 0.5, 2.0, 0.1, 5.0]):
    """
    Compute gradients with adaptive epsilon to handle different sensitivities
    """
    device = naction.device
    B, T, _ = naction.shape
    
    # Convert inputs
    trav_map_np = trav_map.cpu().numpy() if isinstance(trav_map, torch.Tensor) else trav_map
    camera_matrix_np = camera_matrix_orig if isinstance(camera_matrix_orig, np.ndarray) else camera_matrix_orig.cpu().numpy()
    dist_coeffs_np = dist_coeffs if isinstance(dist_coeffs, np.ndarray) else dist_coeffs.cpu().numpy()
    
    def robust_single_traj_score(traj_single):
        """Evaluate single trajectory traversability"""
        try:
            pixel_coords_list = get_traj_pixels_coords(
                camera_matrix_np, dist_coeffs_np, [traj_single], viz_img_size, resize_factor=True
            )
            pixel_coords = pixel_coords_list[0].reshape(-1, 2)
            
            # Clamp to bounds
            pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, trav_map_np.shape[1] - 1)
            pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, trav_map_np.shape[0] - 1)
            
            pixel_coords_int = pixel_coords.astype(int)
            trav_values = trav_map_np[pixel_coords_int[:, 1], pixel_coords_int[:, 0]]
            
            return float(np.mean(trav_values))
            
        except Exception as e:
            print(f"Error in trajectory evaluation: {e}")
            return 0.0
    
    # Compute scores and gradients
    trajs_np = naction.detach().cpu().numpy()
    current_scores = []
    gradients = np.zeros_like(trajs_np)
    gradient_info = []  # For debugging
    
    for b in range(B):
        current_score = robust_single_traj_score(trajs_np[b])
        current_scores.append(current_score)
        
        batch_gradient_info = []
        
        for t in range(T):
            for dim in range(2):  # x, y dimensions
                best_gradient = 0.0
                best_epsilon = base_epsilon
                
                # Try different epsilon values to find meaningful gradient
                for epsilon_scale in epsilon_scales:
                    test_epsilon = base_epsilon * epsilon_scale
                    
                    # Forward difference
                    trajs_plus = trajs_np[b].copy()
                    trajs_plus[t, dim] += test_epsilon
                    score_plus = robust_single_traj_score(trajs_plus)
                    
                    # Backward difference
                    trajs_minus = trajs_np[b].copy()
                    trajs_minus[t, dim] -= test_epsilon
                    score_minus = robust_single_traj_score(trajs_minus)
                    
                    # Central difference (more accurate)
                    gradient_central = (score_plus - score_minus) / (2 * test_epsilon)
                    gradient_forward = (score_plus - current_score) / test_epsilon
                    
                    # Use central difference if both evaluations succeeded
                    if abs(gradient_central) > abs(best_gradient):
                        best_gradient = gradient_central
                        best_epsilon = test_epsilon
                
                gradients[b, t, dim] = best_gradient
                
                batch_gradient_info.append({
                    'batch': b, 'waypoint': t, 'dim': dim,
                    'gradient': best_gradient, 'epsilon': best_epsilon,
                    'score': current_score
                })
        
        gradient_info.extend(batch_gradient_info)
    
    # Convert back to tensors
    current_scores = torch.tensor(current_scores, device=device, dtype=torch.float32)
    traj_grad = torch.tensor(gradients, device=device, dtype=torch.float32)
    
    # Debug info
    total_nonzero_grads = torch.sum(torch.abs(traj_grad) > 1e-8).item()
    grad_norm_per_dim = torch.norm(traj_grad, dim=(0, 1))  # Norm per dimension
    
    print(f"GRADIENT ANALYSIS:")
    print(f"  Total non-zero gradients: {total_nonzero_grads}/{B*T*2}")
    print(f"  X-dimension gradient norm: {grad_norm_per_dim[0].item():.6f}")
    print(f"  Y-dimension gradient norm: {grad_norm_per_dim[1].item():.6f}")
    print(f"  Overall gradient norm: {traj_grad.norm().item():.6f}")
    
    # Check for gradient imbalance
    if grad_norm_per_dim[0] < 0.001 * grad_norm_per_dim[1]:
        print(f"  ⚠️ WARNING: X gradients much smaller than Y gradients")
    elif grad_norm_per_dim[1] < 0.001 * grad_norm_per_dim[0]:
        print(f"  ⚠️ WARNING: Y gradients much smaller than X gradients")
    
    loss = -current_scores.mean()
    
    return loss.item(), traj_grad, gradient_info


def enhanced_traversability_guidance(naction, trav_map, camera_matrix_orig, 
                                    dist_coeffs, viz_img_size, 
                                    min_gradient_norm=1e-6,
                                    gradient_clip=10.0):
    """
    Enhanced version with better gradient handling
    """
    # Compute gradients with adaptive epsilon
    loss, traj_grad, grad_info = adaptive_finite_difference_gradients(
        naction, trav_map, camera_matrix_orig, dist_coeffs, viz_img_size
    )
    
    # Check gradient quality
    grad_norm = traj_grad.norm().item()
    
    if grad_norm < min_gradient_norm:
        print(f"⚠️ Gradient norm too small ({grad_norm:.2e}) - may not provide useful guidance")
        return loss, traj_grad * 0  # Return zero gradients
    
    if grad_norm > gradient_clip:
        print(f"⚠️ Gradient norm too large ({grad_norm:.2e}) - clipping")
        traj_grad = traj_grad * (gradient_clip / grad_norm)
    
    # Check for gradient sparsity
    nonzero_grads = torch.sum(torch.abs(traj_grad) > 1e-8).item()
    total_grads = traj_grad.numel()
    sparsity = nonzero_grads / total_grads
    
    print(f"Gradient sparsity: {sparsity:.3f} ({nonzero_grads}/{total_grads} non-zero)")
    
    if sparsity < 0.1:
        print(f"⚠️ Very sparse gradients - guidance may be weak")
    
    return loss, traj_grad


def visualize_trajectory_gradients(naction, traj_grad, trav_map):
    """
    Visualize gradients on the trajectory for debugging
    """
    import matplotlib.pyplot as plt
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot traversability map with trajectory
        ax = axes[0, 0]
        ax.imshow(trav_map, cmap='viridis', origin='upper', alpha=0.8)
        ax.set_title('Traversability Map + Trajectory')
        
        # Get pixel coordinates for first trajectory
        first_traj = naction[0].detach().cpu().numpy()
        try:
            camera_matrix_np = camera_matrix_orig.cpu().numpy() if hasattr(camera_matrix_orig, 'cpu') else camera_matrix_orig
            dist_coeffs_np = dist_coeffs.cpu().numpy() if hasattr(dist_coeffs, 'cpu') else dist_coeffs
            
            pixel_coords_list = get_traj_pixels_coords(
                camera_matrix_np, dist_coeffs_np, [first_traj], VIZ_IMAGE_SIZE_FISHEYE, resize_factor=True
            )
            pixel_coords = pixel_coords_list[0].reshape(-1, 2)
            
            ax.plot(pixel_coords[:, 0], pixel_coords[:, 1], 'r-', linewidth=2, alpha=0.8)
            ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c='red', s=30)
        except Exception as e:
            print(f"Could not overlay trajectory: {e}")
        
        # Plot gradient magnitudes
        ax = axes[0, 1]
        grad_magnitude = torch.norm(traj_grad, dim=-1).cpu().numpy()  # [B, T]
        im = ax.imshow(grad_magnitude, cmap='hot', aspect='auto')
        ax.set_title('Gradient Magnitudes')
        ax.set_xlabel('Waypoint')
        ax.set_ylabel('Trajectory (Batch)')
        plt.colorbar(im, ax=ax)
        
        # Plot X gradients
        ax = axes[1, 0]
        x_grads = traj_grad[:, :, 0].cpu().numpy()
        im = ax.imshow(x_grads, cmap='RdBu', aspect='auto', vmin=-np.abs(x_grads).max(), vmax=np.abs(x_grads).max())
        ax.set_title('X Gradients')
        ax.set_xlabel('Waypoint')
        ax.set_ylabel('Trajectory (Batch)')
        plt.colorbar(im, ax=ax)
        
        # Plot Y gradients
        ax = axes[1, 1]
        y_grads = traj_grad[:, :, 1].cpu().numpy()
        im = ax.imshow(y_grads, cmap='RdBu', aspect='auto', vmin=-np.abs(y_grads).max(), vmax=np.abs(y_grads).max())
        ax.set_title('Y Gradients')
        ax.set_xlabel('Waypoint')
        ax.set_ylabel('Trajectory (Batch)')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('gradient_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Gradient visualization saved to 'gradient_analysis.png'")
        
    except Exception as e:
        print(f"Gradient visualization failed: {e}")


# Updated main loop integration
def apply_enhanced_guidance(naction, trav_map, camera_matrix_orig, dist_coeffs, viz_img_size, step_idx):
    """
    Apply enhanced guidance with comprehensive monitoring
    """
    print(f"\n--- Traversability Guidance Step {step_idx} ---")
    
    # Compute enhanced gradients
    trav_loss, traj_grad = enhanced_traversability_guidance(
        naction, trav_map, camera_matrix_orig, dist_coeffs, viz_img_size
    )
    
    # Visualize gradients occasionally
    if step_idx % 10 == 0:  # Every 10 steps
        visualize_trajectory_gradients(naction, traj_grad, trav_map)
    
    grad_norm = traj_grad.norm().item()
    
    if grad_norm > 1e-6:
        # Apply time-dependent scaling
        timestep_ratio = step_idx / 50  # Assuming 50 total steps, adjust as needed
        grad_scale = 1.0 * (1 - timestep_ratio) * min(1.0, grad_norm)  # Scale down if gradients are too large
        
        print(f"Applying guidance: loss={trav_loss:.4f}, grad_norm={grad_norm:.6f}, scale={grad_scale:.3f}")
        
        naction_new = naction + grad_scale * traj_grad
        
        # Evaluate improvement
        with torch.no_grad():
            old_loss, _ = enhanced_traversability_guidance(naction, trav_map, camera_matrix_orig, dist_coeffs, viz_img_size)
            new_loss, _ = enhanced_traversability_guidance(naction_new, trav_map, camera_matrix_orig, dist_coeffs, viz_img_size)
            
        improvement = old_loss - new_loss
        print(f"Expected improvement: {improvement:.6f}")
        
        return naction_new
    else:
        print("❌ Skipping guidance - gradients too small")
        return naction






####################




import numpy as np
import torch
import torch.nn.functional as F

def smooth_traversability_guidance(naction, trav_map, camera_matrix_orig, 
                                 dist_coeffs, viz_img_size, 
                                 guidance_strength=0.1,  # Much smaller!
                                 smoothing_weight=0.8,   # Preserve original direction
                                 max_displacement=0.05): # Maximum allowed change per waypoint
    """
    Smooth traversability guidance that gently nudges trajectories
    while preserving their original structure and direction
    """
    device = naction.device
    B, T, _ = naction.shape
    
    print(f"Applying SMOOTH guidance (strength={guidance_strength:.3f})")
    
    # Store original trajectory for smoothing
    naction_original = naction.clone()
    
    # Convert inputs
    trav_map_np = trav_map.cpu().numpy() if isinstance(trav_map, torch.Tensor) else trav_map
    camera_matrix_np = camera_matrix_orig if isinstance(camera_matrix_orig, np.ndarray) else camera_matrix_orig.cpu().numpy()
    dist_coeffs_np = dist_coeffs if isinstance(dist_coeffs, np.ndarray) else dist_coeffs.cpu().numpy()
    
    def evaluate_trajectory_score(traj_single):
        """Clean trajectory evaluation"""
        try:
            pixel_coords_list = get_traj_pixels_coords(
                camera_matrix_np, dist_coeffs_np, [traj_single], viz_img_size, resize_factor=True
            )
            pixel_coords = pixel_coords_list[0].reshape(-1, 2)
            
            # Clamp to bounds
            pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, trav_map_np.shape[1] - 1)
            pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, trav_map_np.shape[0] - 1)
            
            pixel_coords_int = pixel_coords.astype(int)
            trav_values = trav_map_np[pixel_coords_int[:, 1], pixel_coords_int[:, 0]]
            
            return float(np.mean(trav_values))
        except:
            return 0.0
    
    # Compute gentle gradients with small epsilon
    trajs_np = naction.detach().cpu().numpy()
    gradients = np.zeros_like(trajs_np)
    current_scores = []
    
    # Use smaller epsilon for more stable gradients
    epsilon = 0.01  # Smaller step for more stable gradients
    
    for b in range(B):
        current_score = evaluate_trajectory_score(trajs_np[b])
        current_scores.append(current_score)
        
        # Compute gradients only for a subset of waypoints to maintain smoothness
        for t in range(T):
            for dim in range(2):
                # Forward difference with small epsilon
                trajs_perturbed = trajs_np[b].copy()
                trajs_perturbed[t, dim] += epsilon
                
                perturbed_score = evaluate_trajectory_score(trajs_perturbed)
                raw_gradient = (perturbed_score - current_score) / epsilon
                
                # Apply gradient smoothing and clipping
                gradients[b, t, dim] = np.clip(raw_gradient, -10.0, 10.0)  # Clip individual gradients
    
    # Convert to tensors
    current_scores = torch.tensor(current_scores, device=device, dtype=torch.float32)
    traj_grad = torch.tensor(gradients, device=device, dtype=torch.float32)
    
    # Apply spatial smoothing to gradients (smooth across waypoints)
    traj_grad_smoothed = apply_spatial_smoothing(traj_grad)
    
    # Scale gradients appropriately
    grad_norm = traj_grad_smoothed.norm().item()
    # print(f"Raw gradient norm: {grad_norm:.4f}")
    
    if grad_norm > 0.001:  # Only apply if meaningful gradients exist
        # Normalize gradients to have reasonable magnitude
        max_allowed_norm = 1.0  # Maximum gradient norm
        if grad_norm > max_allowed_norm:
            traj_grad_smoothed = traj_grad_smoothed * (max_allowed_norm / grad_norm)
            # print(f"Normalized gradient norm: {traj_grad_smoothed.norm().item():.4f}")
        
        # Apply gentle guidance
        guidance_update = guidance_strength * traj_grad_smoothed
        
        # Limit per-waypoint displacement
        displacement_norm = torch.norm(guidance_update, dim=-1, keepdim=True)  # [B, T, 1]
        displacement_scale = torch.clamp(displacement_norm / max_displacement, min=1.0)
        guidance_update = guidance_update / displacement_scale
        
        # print(f"Max waypoint displacement: {torch.norm(guidance_update, dim=-1).max().item():.4f}")
        
        # Blend with original trajectory (preserve structure)
        naction_guided = naction_original + guidance_update
        naction_final = smoothing_weight * naction_original + (1 - smoothing_weight) * naction_guided
        
        # Compute actual improvement
        old_score = current_scores.mean().item()
        
        # Evaluate new trajectory
        with torch.no_grad():
            new_scores = []
            for b in range(B):
                new_traj = naction_final[b].detach().cpu().numpy()
                new_score = evaluate_trajectory_score(new_traj)
                new_scores.append(new_score)
            new_scores = torch.tensor(new_scores, device=device)
            new_score_mean = new_scores.mean().item()
        
        improvement = new_score_mean - old_score
        print(f"Traversability: {old_score:.4f} → {new_score_mean:.4f} (Δ={improvement:.4f})")
        
        if improvement > 0:
            print("✅ Improvement achieved")
            return naction_final
        else:
            print("⚠️ No improvement, keeping original")
            return naction_original
    
    else:
        print("❌ Gradients too small, keeping original")
        return naction_original


def apply_spatial_smoothing(traj_grad, kernel_size=3):
    """
    Apply spatial smoothing to gradients along the trajectory
    to ensure smooth transitions between waypoints
    """
    B, T, D = traj_grad.shape
    
    if T < kernel_size:
        return traj_grad
    
    # Apply 1D convolution for smoothing along time dimension
    traj_grad_smooth = traj_grad.clone()
    
    for d in range(D):  # For each dimension (x, y)
        # Reshape for conv1d: [B, 1, T]
        grad_d = traj_grad[:, :, d].unsqueeze(1)
        
        # Create smoothing kernel
        kernel = torch.ones(1, 1, kernel_size, device=traj_grad.device) / kernel_size
        
        # Apply padding and convolution
        grad_d_padded = F.pad(grad_d, (kernel_size//2, kernel_size//2), mode='reflect')
        grad_d_smooth = F.conv1d(grad_d_padded, kernel)
        
        traj_grad_smooth[:, :, d] = grad_d_smooth.squeeze(1)
    
    return traj_grad_smooth


def progressive_guidance_schedule(step_idx, total_steps):
    """
    Progressive guidance that starts very gentle and gradually increases
    """
    progress = step_idx / total_steps
    
    if progress < 0.3:  # First 30% - very gentle
        return 0.02
    elif progress < 0.6:  # Middle 30% - moderate
        return 0.05
    else:  # Final 40% - stronger
        return 0.1


def apply_smooth_traversability_guidance(naction, trav_map, camera_matrix_orig, 
                                       dist_coeffs, viz_img_size, step_idx, total_steps=50):
    """
    Main function to apply smooth guidance with progressive scheduling
    """
    print(f"\n--- Smooth Traversability Guidance Step {step_idx}/{total_steps} ---")
    
    # Progressive guidance strength
    guidance_strength = progressive_guidance_schedule(step_idx, total_steps)
    
    # Adjust smoothing based on progress (more preservation early on)
    progress = step_idx / total_steps
    smoothing_weight = 0.95 - 0.2 * progress  # Start at 0.95, end at 0.75
    
    # Apply smooth guidance
    naction_new = smooth_traversability_guidance(
        naction, trav_map, camera_matrix_orig, dist_coeffs, viz_img_size,
        guidance_strength=guidance_strength,
        smoothing_weight=smoothing_weight,
        max_displacement=0.03  # Small displacement limit
    )
    
    # Compute trajectory smoothness metric
    trajectory_smoothness = compute_trajectory_smoothness(naction_new)
    print(f"Trajectory smoothness: {trajectory_smoothness:.4f}")
    
    return naction_new


def compute_trajectory_smoothness(trajectories):
    """
    Compute smoothness metric for trajectories
    Lower values = smoother trajectories
    """
    # Compute second derivatives (acceleration)
    if trajectories.shape[1] < 3:
        return 0.0
    
    # First derivative (velocity)
    velocity = trajectories[:, 1:] - trajectories[:, :-1]  # [B, T-1, 2]
    
    # Second derivative (acceleration)
    acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, T-2, 2]
    
    # Smoothness is inverse of acceleration magnitude
    smoothness = torch.mean(torch.norm(acceleration, dim=-1))
    
    return smoothness.item()


def visualize_guidance_effect(naction_before, naction_after, trav_map, step_idx):
    """
    Visualize the effect of guidance on trajectories
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot traversability map
        for ax in axes:
            ax.imshow(trav_map, cmap='viridis', origin='upper', alpha=0.7)
        
        # Function to get pixel coords
        def get_pixel_coords_safe(traj):
            try:
                camera_matrix_np = camera_matrix_orig.cpu().numpy() if hasattr(camera_matrix_orig, 'cpu') else camera_matrix_orig
                dist_coeffs_np = dist_coeffs.cpu().numpy() if hasattr(dist_coeffs, 'cpu') else dist_coeffs
                
                pixel_coords_list = get_traj_pixels_coords(
                    camera_matrix_np, dist_coeffs_np, [traj], VIZ_IMAGE_SIZE_FISHEYE, resize_factor=True
                )
                return pixel_coords_list[0].reshape(-1, 2)
            except:
                return None
        
        # Plot first trajectory before guidance
        traj_before = naction_before[0].detach().cpu().numpy()
        pixel_coords_before = get_pixel_coords_safe(traj_before)
        if pixel_coords_before is not None:
            axes[0].plot(pixel_coords_before[:, 0], pixel_coords_before[:, 1], 'r-', linewidth=2, alpha=0.8, label='Before')
            axes[0].scatter(pixel_coords_before[:, 0], pixel_coords_before[:, 1], c='red', s=30, alpha=0.8)
        axes[0].set_title('Before Guidance')
        axes[0].legend()
        
        # Plot first trajectory after guidance
        traj_after = naction_after[0].detach().cpu().numpy()
        pixel_coords_after = get_pixel_coords_safe(traj_after)
        if pixel_coords_after is not None:
            axes[1].plot(pixel_coords_after[:, 0], pixel_coords_after[:, 1], 'b-', linewidth=2, alpha=0.8, label='After')
            axes[1].scatter(pixel_coords_after[:, 0], pixel_coords_after[:, 1], c='blue', s=30, alpha=0.8)
        axes[1].set_title('After Guidance')
        axes[1].legend()
        
        # Plot both trajectories for comparison
        if pixel_coords_before is not None:
            axes[2].plot(pixel_coords_before[:, 0], pixel_coords_before[:, 1], 'r-', linewidth=2, alpha=0.6, label='Before')
        if pixel_coords_after is not None:
            axes[2].plot(pixel_coords_after[:, 0], pixel_coords_after[:, 1], 'b-', linewidth=2, alpha=0.8, label='After')
        axes[2].set_title('Comparison')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(f'guidance_effect_step_{step_idx:03d}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Guidance visualization saved to 'guidance_effect_step_{step_idx:03d}.png'")
        
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")


# Usage in your main sampling loop - REPLACE your guidance section with this:
def your_updated_sampling_loop(model, noise_scheduler, naction, obs_cond,
                               trav_img_subscriber, camera_matrix_orig, dist_coeffs, args):
    """
    Updated sampling loop with smooth guidance
    """
    for i, k in enumerate(noise_scheduler.timesteps):
        print(f"\nDiffusion step {i+1}/{len(noise_scheduler.timesteps)}")
        
        # 1. Diffusion step
        with torch.no_grad():
            noise_pred = model(
                'noise_pred_net',
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # 2. SMOOTH traversability guidance
        if args.enable_trav:
            trav_map = trav_img_subscriber.get_trav_img()
            
            # Store before state for visualization
            naction_before = naction.clone()
            
            # Apply smooth guidance
            naction = apply_smooth_traversability_guidance(
                naction, trav_map, camera_matrix_orig, dist_coeffs, 
                VIZ_IMAGE_SIZE_FISHEYE, i, len(noise_scheduler.timesteps)
            )
            
            # Visualize effect occasionally
            if i % 10 == 0:
                visualize_guidance_effect(naction_before, naction, trav_map, i)
    
    return naction