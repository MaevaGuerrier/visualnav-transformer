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

# def notwork_compute_traversability_scores(
#     trajs: list,
#     trav_img: np.ndarray,
#     camera_matrix: np.ndarray,
#     dist_coeffs: np.ndarray,
#     viz_img_size: Tuple[int, int],
#     resize_factor: bool = True,
#     device: torch.device = torch.device("cpu"),
# ) -> np.ndarray:
#     """
#     Compute the traversability scores for a list of trajectories.
#     """
#     traj_pix_coords = get_traj_pixels_coords(
#         camera_matrix, dist_coeffs, trajs, viz_img_size, resize_factor=resize_factor
#     ) # returns a list
#     traj_pix_coords = np.array(traj_pix_coords, dtype=np.float32)
#     traj_pix_coords = torch.from_numpy(traj_pix_coords).float().to(device) # .to(device) creates a new tensor 
#     traj_pix_coords.requires_grad_()  # which is why we enable grad after moving to device
#     traj_pix_coords = traj_pix_coords.squeeze(2)  # [B, T, 2]

#     print("traj_pix_coords:", traj_pix_coords.requires_grad, traj_pix_coords.grad_fn)


#     # [1, 1, H, W] grid_sample https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
#     traversability_map = torch.tensor(trav_img, dtype=torch.float32, device=device)
#     traversability_map = traversability_map.unsqueeze(0).unsqueeze(0)
   
#     B, _, _ = traj_pix_coords.shape
#     H, W = traversability_map.shape[-2:]

#     # normalize to [-1, 1] for grid_sample
#     x = (traj_pix_coords[..., 0] / (W - 1)) * 2 - 1
#     y = (traj_pix_coords[..., 1] / (H - 1)) * 2 - 1
#     coords = torch.stack([x, y], dim=-1).unsqueeze(1)
#     coords.requires_grad_()

#     print("coords", coords.requires_grad, coords.grad_fn)

#     # sample traversability
#     values = F.grid_sample(
#         traversability_map.expand(B, -1, -1, -1),  # [B,1,H,W]
#         coords,  # [B,1,T,2]
#         align_corners=True
#     )

#     # result: [B,1,1,T]
#     values = values.squeeze(1).squeeze(1)  # [B,T]

#     traj_score = values.mean(dim=1)  # [B]
#     score = traj_score.mean() 

#     print("--------------------------------")
#     print("score", score.requires_grad, score.grad_fn)
#     exit()
#     grad = torch.autograd.grad(score, traj_pix_coords, create_graph=False)[0]

#     return grad

# def compute_traversability_scores(
#     trajs: list,
#     trav_img: np.ndarray,
#     camera_matrix: np.ndarray,
#     dist_coeffs: np.ndarray,
#     viz_img_size: Tuple[int, int],
#     resize_factor: bool = True,
#     device: torch.device = torch.device("cpu"),
# ) -> np.ndarray:
#     """
#     Compute differentiable traversability gradients w.r.t. trajectory pixel coordinates.
    
#     Args:
#         trajs: list or array of trajectories (B, T, 2) in world coordinates
#         traversability_img: H x W numpy image with traversability values
#         camera_matrix, dist_coeffs: camera intrinsics for projection
#         viz_img_size: image size for visualization
#         device: 'cuda' or 'cpu'
#         resize_factor: optional scaling factor
    
#     Returns:
#         grad: tensor of shape [B, T, 2], gradients of the score w.r.t. traj_pix_coords
#     """

#     # -------------------------------
#     # 1. Get trajectory pixel coordinates
#     # -------------------------------
#     traj_pix_coords = get_traj_pixels_coords(
#         camera_matrix, dist_coeffs, trajs, viz_img_size, resize_factor=resize_factor
#     )  # returns list of shape [B, T, 1, 2] or similar

#     # Convert to PyTorch tensor and ensure grad tracking
#     traj_pix_coords = np.array(traj_pix_coords, dtype=np.float32)
#     traj_pix_coords = torch.from_numpy(traj_pix_coords).to(device).requires_grad_()
#     print("traj_pix_coords:", traj_pix_coords.requires_grad, traj_pix_coords.grad_fn)

#     # Remove singleton dim if present, final shape: [B, T, 2]
#     traj_pix_coords = traj_pix_coords.squeeze(2)
#     B, _, _ = traj_pix_coords.shape

#     # -------------------------------
#     # 2. Prepare traversability map
#     # -------------------------------
#     traversability_map = torch.tensor(trav_img, dtype=torch.float32, device=device)
#     traversability_map = traversability_map.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
#     H, W = traversability_map.shape[-2:]
#     print("type of H W", type(H), type(W))
#     W = torch.tensor(W, dtype=torch.float32, device=device)
#     H = torch.tensor(H, dtype=torch.float32, device=device)

#     # -------------------------------
#     # 3. Normalize trajectory coords to [-1, 1] for grid_sample
#     # -------------------------------
#     x = (traj_pix_coords[..., 0] / (W - 1)) * 2 - 1  # [B,T]
#     y = (traj_pix_coords[..., 1] / (H - 1)) * 2 - 1  # [B,T]
#     coords = torch.stack([x, y], dim=-1).unsqueeze(2)  # [B,1,T,2]

#     coords.requires_grad_()
#     print("coords", coords.requires_grad, coords.grad_fn)

#     # -------------------------------
#     # 4. Sample traversability values along trajectory
#     # -------------------------------
#     # F.grid_sample(input, grid) grad w.r.t grid only if grid has grad
#     # In our case we don't want grad w.r.t traversability_map only for traj
#     print("coord shape", coords.shape)
#     values = F.grid_sample(
#         traversability_map.expand(B, -1, -1, -1),  # [B,1,H,W]
#         coords,                                   # [B,1,T,2]
#         align_corners=True,
#         mode="bicubic"
#     )

#     # values: [B,1,1,T] -> squeeze to [B,T]
#     values = values.squeeze(1).squeeze(1)
#     print("values", values.requires_grad, values.grad_fn)

#     # -------------------------------
#     # 5. Compute trajectory score
#     # -------------------------------
#     traj_score = values.mean(dim=1)  # mean per trajectory
#     score = traj_score.mean()        # mean over batch

#     # -------------------------------
#     # 6. Compute gradient w.r.t traj_pix_coords
#     # -------------------------------
#     print("score.requires_grad:", score.requires_grad, "score.grad_fn:", score.grad_fn)
#     exit()
#     grad = torch.autograd.grad(score, traj_pix_coords, create_graph=False)[0]  # [B,T,2]

#     return grad


# TODO FIGURE IT OUT
# For now correcting only one traj for debugging
# def get_gradient_traversability(trajs, trav_img, fully_traversable_value=1.0):

#     if trav_img is None:
#         return 
    
#     # traversability image, 0..1 values
#     traversability_img = torch.tensor(trav_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

#     # F.grid_sample -> grid specifies the sampling pixel locations normalized by the input spatial dimensions. 
#     # Therefore, it should have most values in the range of [-1, 1]
#     # see https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
#     traj_normalized = torch.tensor(trajs.copy(), dtype=torch.float32, requires_grad=True)  # [N,2]
#     rospy.logdebug(f"Trajectory before traversability correction: {traj_normalized}")
    
#     # grid for sampling: [1,N,1,2]
#     grid = traj_normalized.unsqueeze(0).unsqueeze(2)

#     traversability_vals = F.grid_sample(traversability_img, grid, align_corners=True)
#     traversability_vals = traversability_vals.squeeze()  
#     rospy.logdebug(f"Traversability values along the trajectory: {traversability_vals}")

#     cost = torch.sum(fully_traversable_value - traversability_vals)
#     cost.backward()
#     rospy.logdebug(f"Trajectory gradient after traversability correction: {traj_normalized.grad}")
#     rospy.logdebug(f"Cost value: {cost.item()}")

#     return traj_normalized.grad

# #         # with torch.no_grad():
# #         #     traj_normalized -= lr * traj.grad
# #         #     traj_normalized.grad.zero_()



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
    
    # print(f"Gradient norm: {gradients.norm().item():.6f}")
    # print(f"Traversability loss: {loss.item():.6f}")
    
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




