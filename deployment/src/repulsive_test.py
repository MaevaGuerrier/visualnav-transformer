import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, gaussian_filter
import cv2

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, gaussian_filter
from scipy.interpolate import interp1d
import yaml
from trav_utils import get_traj_pixels_coords
from typing import Tuple

# CONSTANTS
ROBOT_CONFIG_PATH ="../config/robot.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]

class RepulsiveFieldPlanner:
    def __init__(self, 
                 robot_radius_pixels,
                 max_linear_vel,
                 max_angular_vel,
                 repulsion_gain=1.0,
                 safety_margin_pixels=5,
                 influence_radius_pixels=30):
        """
        Initialize repulsive field planner.
        
        Args:
            robot_radius_pixels: Robot radius in pixel space
            max_linear_vel: Maximum linear velocity (m/s)
            max_angular_vel: Maximum angular velocity (rad/s)
            repulsion_gain: Strength of repulsive force
            safety_margin_pixels: Additional safety buffer beyond robot radius
            influence_radius_pixels: Distance at which repulsion begins
        """
        self.robot_radius = robot_radius_pixels
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.repulsion_gain = repulsion_gain
        self.safety_margin = safety_margin_pixels
        self.influence_radius = influence_radius_pixels
        
        self.distance_field = None
        self.gradient_x = None
        self.gradient_y = None
    
    def create_distance_field(self, traversability_map, threshold=0.5):
        """
        Create distance field from traversability map.
        
        Args:
            traversability_map: 2D array where 0=untraversable, 1=fully traversable
            threshold: Traversability threshold below which is considered obstacle
            
        Returns:
            distance_field: Distance to nearest obstacle for each pixel
        """
        # Identify obstacles
        obstacles = traversability_map < threshold
        
        # Dilate obstacles by robot radius + safety margin
        structure_size = int(2 * (self.robot_radius + self.safety_margin) + 1)
        structure = np.ones((structure_size, structure_size))
        dilated_obstacles = binary_dilation(obstacles, structure=structure)
        
        # Compute distance transform (distance to nearest obstacle)
        free_space = ~dilated_obstacles
        distance_field = distance_transform_edt(free_space)
        
        # Smooth the distance field for better gradients
        distance_field = gaussian_filter(distance_field, sigma=2.0)
        
        self.distance_field = distance_field
        
        # Compute gradients for repulsive force direction
        self.gradient_y, self.gradient_x = np.gradient(distance_field)
        
        return distance_field
    

    def compute_repulsive_force(self, trajs):
        """
        Compute pointwise repulsive forces for trajectories.
        
        Args:
            trajs: (n_traj, n_points, 2) or (n_points, 2) pixel coordinates

        Returns:
            repulsion: same shape as trajs, forces (fx, fy) per point
        """
        arr = np.asarray(trajs, dtype=float)
        single = False
        if arr.ndim == 2:  # (n_points, 2)
            arr = arr[None, ...]
            single = True

        h, w = self.distance_field.shape

        # Clamp to valid pixel indices
        xs = np.clip(arr[..., 0].astype(int), 0, w - 1)
        ys = np.clip(arr[..., 1].astype(int), 0, h - 1)

        # Distances and gradients
        dist = self.distance_field[ys, xs]
        gx = self.gradient_x[ys, xs]
        gy = self.gradient_y[ys, xs]

        # Normalize gradients
        grad_mag = np.sqrt(gx**2 + gy**2) + 1e-6
        gx /= grad_mag
        gy /= grad_mag

        # Repulsion magnitude (zero outside influence radius)
        dist_safe = np.maximum(dist, 1.0)  # avoid div by zero
        rep_mag = self.repulsion_gain * (1.0 / dist_safe - 1.0 / self.influence_radius)
        rep_mag[dist > self.influence_radius] = 0.0

        fx = rep_mag * gx
        fy = rep_mag * gy

        repulsion = np.stack([fx, fy], axis=-1)

        return repulsion[0] if single else repulsion



    def modify_trajectory(self, trajectory_pixels, traversability_map, camera_matrix, dist_coeffs, viz_img_size, dt):
        """
        Modify trajectory using repulsive field while respecting dynamics.
        
        Args:
            trajectory_pixels: Array of shape (N, 2) OR (M, N, 2)
                             Single trajectory: (N, 2) with (x, y) pixel coordinates
                             Multiple trajectories: (M, N, 2) where M=num trajectories
            dt: Time step between waypoints (seconds)
            
        Returns:
            modified_trajectory: Safe trajectory in pixel space (same shape as input)
            safety_scores: Safety metric for each waypoint (0=unsafe, 1=safe)
                          Shape (N,) for single traj or (M, N) for multiple
        """
        # Handle both single and batch trajectories
        is_batch = len(trajectory_pixels.shape) == 3
        
        if is_batch:
            return self._modify_trajectory_batch(trajectory_pixels, traversability_map, camera_matrix, dist_coeffs, viz_img_size, dt)
        else:
            return self._modify_trajectory_single(trajectory_pixels, traversability_map, camera_matrix, dist_coeffs, viz_img_size, dt)
    


    def _modify_trajectory_single(self, traj, traversability_map, camera_matrix, dist_coeffs, viz_img_size, dt):
        """
        Modify a single trajectory with repulsion in pixel space,
        then apply kinematic constraints in world space, then back to pixel space.
        """

        # Step 1: Repulsion (still in pixel space)
        repulsion = self.compute_repulsive_force(traj)
        traj = traj + repulsion

        # Step 2: Convert to world coordinates (meters)
        traj_world = self.pixels_to_world(
            traj[None], camera_matrix, dist_coeffs, viz_img_size
        )[0]  # shape (N,2)

        # Step 3: Apply kinematic constraints in world space
        traj_world = self.apply_kinematic_constraints_world(
            traj_world, dt, self.max_linear_vel
        )

        traj_pixels = get_traj_pixels_coords(
            traj_world, camera_matrix, dist_coeffs, viz_img_size
        )

        return traj_pixels


    def _modify_trajectory_batch(self, trajs, traversability_map, camera_matrix, dist_coeffs, viz_img_size, dt):
        """
        Same as single, but for a batch of trajectories.
        """

        # Step 1: Repulsion (pixel space)
        repulsion = self.compute_repulsive_force(trajs)
        trajs = trajs + repulsion

        trajs_world = self.get_world_coords_from_pixels(
                pixel_coords=trajs,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                viz_img_size=viz_img_size,
                camera_height=0.25,
                camera_x_offset=0.10,
                resize_factor=True
            )

        # Step 3: Apply kinematic constraints in world space
        trajs_world = self.apply_kinematic_constraints_world(
            trajs_world, dt, self.max_linear_vel
        )

        traj_pixels = get_traj_pixels_coords(
                        camera_matrix, dist_coeffs, trajs_world, viz_img_size
                    )
        
        return traj_pixels



    def _smooth_trajectory(self, trajectory, window=3):
        """Apply moving average smoothing."""
        if len(trajectory) < window:
            return trajectory
        
        smoothed = trajectory.copy()
        for i in range(1, len(trajectory) - 1):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(trajectory), i + window // 2 + 1)
            smoothed[i] = np.mean(trajectory[start_idx:end_idx], axis=0)
        
        return smoothed
    
    def _apply_kinematic_constraints(self, trajectory, dt):
        """Ensure trajectory respects velocity limits."""
        constrained = trajectory.copy()
        
        for i in range(1, len(trajectory)):
            dx = constrained[i, 0] - constrained[i-1, 0]
            dy = constrained[i, 1] - constrained[i-1, 1]
            
            # This is a simplified constraint in pixel space
            # You may need to scale by your pixel-to-meter conversion
            displacement = np.sqrt(dx**2 + dy**2)
            max_displacement = self.max_linear_vel * dt  # Assume 1 pixel = some meters
            
            if displacement > max_displacement:
                scale = max_displacement / displacement
                constrained[i] = constrained[i-1] + scale * np.array([dx, dy])
        
        return constrained
    

    def apply_kinematic_constraints_world(self, trajectories_world, dt, max_linear_vel):
        """
        trajectories_world: (n_traj, n_points, 2) in meters
        returns constrained trajectories of same shape
        """
        trajs = np.array(trajectories_world, dtype=float)
        single = False
        if trajs.ndim == 2:  # (n_points,2)
            trajs = trajs[None, ...]
            single = True

        n_traj, n_points, _ = trajs.shape
        constrained = trajs.copy()
        max_disp_m = max_linear_vel * dt  # meters

        for i in range(1, n_points):
            dx = constrained[:, i, 0] - constrained[:, i-1, 0]  # (n_traj,)
            dy = constrained[:, i, 1] - constrained[:, i-1, 1]
            disp = np.sqrt(dx**2 + dy**2)

            mask = disp > max_disp_m
            # compute scale only where needed
            scale = np.ones_like(disp)
            scale[mask] = max_disp_m / disp[mask]

            constrained[:, i, 0] = constrained[:, i-1, 0] + scale * dx
            constrained[:, i, 1] = constrained[:, i-1, 1] + scale * dy

        return constrained[0] if single else constrained



    def _compute_safety_scores(self, trajectory):
        """Compute safety score for each waypoint."""
        scores = np.zeros(len(trajectory))
        
        for i, point in enumerate(trajectory):
            x_int = int(np.clip(point[0], 0, self.distance_field.shape[1] - 1))
            y_int = int(np.clip(point[1], 0, self.distance_field.shape[0] - 1))
            
            dist = self.distance_field[y_int, x_int]
            
            # Score based on distance: 1.0 if far from obstacles, 0.0 if at obstacle
            scores[i] = np.clip(dist / self.influence_radius, 0.0, 1.0)
        
        return scores
    
    def visualize_field(self, traversability_map):
        """
        Create visualization arrays for the repulsive field.
        
        Returns:
            field_viz: Dictionary with visualization data
        """
        distance_field = self.create_distance_field(traversability_map)
        
        # Create potential field visualization
        h, w = distance_field.shape
        potential = np.zeros((h, w))
        
        for y in range(h):
            for x in range(w):
                dist = distance_field[y, x]
                if dist < self.influence_radius:
                    potential[y, x] = self.repulsion_gain * (1.0 / (dist + 1e-6) - 1.0 / self.influence_radius)
        
        return {
            'distance_field': distance_field,
            'potential_field': potential,
            'gradient_x': self.gradient_x,
            'gradient_y': self.gradient_y
        }

    def unproject_pixels(
        self,
        uv: np.ndarray,
        camera_height: float,
        camera_x_offset: float,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> np.ndarray:
        """
        Unprojects 2D pixel coordinates to 3D world coordinates on the ground plane (z=0).
        
        Args:
            uv: array of shape (N, 2) or (N, 1, 2) representing (u, v) pixel coordinates
            camera_height: height of camera above ground plane
            camera_x_offset: x offset of camera
            camera_matrix: camera intrinsic matrix
            dist_coeffs: distortion coefficients
            
        Returns:
            xy: array of shape (N, 2) representing (x, y) world coordinates
        """
        # Handle different input shapes
        original_shape = uv.shape
        if len(uv.shape) == 3:
            # Shape is (N, 1, 2) or similar
            uv = uv.reshape(-1, 2)
        
        N = uv.shape[0]
        
        # Undistort the pixel coordinates
        if dist_coeffs is None:
            uv_undistorted = cv2.undistortPoints(
                uv.reshape(N, 1, 2).astype(np.float64),
                camera_matrix,
                dist_coeffs,
                P=camera_matrix
            ).reshape(N, 2)
        else:
            uv_undistorted = cv2.fisheye.undistortPoints(
                uv.reshape(N, 1, 2).astype(np.float64),
                camera_matrix,
                dist_coeffs,
                P=camera_matrix
            ).reshape(N, 2)
        
        # Convert to normalized camera coordinates
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        x_norm = (uv_undistorted[:, 0] - cx) / fx
        y_norm = (uv_undistorted[:, 1] - cy) / fy
        
        # Ray direction in camera frame: [x_norm, y_norm, 1]
        # The camera frame from your code is: (y_cv, -z_cv, x_cv) = (y_world, -z_world, x_world)
        # So: x_cv = x_world, y_cv = y_world, z_cv = -z_world
        
        # In camera coordinates, the ray is [x_norm, y_norm, 1]
        # This corresponds to world coordinates: [1, x_norm, -y_norm] (scaled)
        
        # Ground plane constraint: z_world = 0
        # Camera is at z = camera_height
        # Ray: (x_w, y_w, z_w) = camera_pos + t * direction
        # z_world = camera_height + t * (-y_norm) = 0
        # Therefore: t = camera_height / y_norm
        
        t = camera_height / (-y_norm)
        
        # Compute world coordinates
        # x_world direction corresponds to z_cv = 1 in camera frame
        # y_world direction corresponds to x_cv = x_norm in camera frame
        x_world = t * 1.0 - camera_x_offset
        y_world = t * x_norm
        
        xy = np.stack([x_world, y_world], axis=-1)
        
        # Reshape back to match original shape if needed
        if len(original_shape) == 3:
            xy = xy.reshape(original_shape[0], original_shape[1], 2)
        
        return xy
    

    def get_world_coords_from_pixels(
        self,
        pixel_coords: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        viz_img_size: Tuple[int, int],
        camera_height: float = 0.25,
        camera_x_offset: float = 0.10,
        resize_factor: bool = True
    ) -> np.ndarray:
        """
        Converts pixel coordinates back to world/robot frame (x, y) coordinates.
        
        Args:
            pixel_coords: array of shape (N, 2) or (N, 1, 2) representing (u, v) pixel coordinates
            camera_matrix: camera intrinsic matrix
            dist_coeffs: distortion coefficients
            viz_img_size: (width, height) of the visualization image
            camera_height: height of camera in meters
            camera_x_offset: x offset of camera in meters
            resize_factor: whether resize factor was applied during forward projection
            
        Returns:
            xy_coords: array of shape matching input representing (x, y) world coordinates
        """
        original_shape = pixel_coords.shape
        pixel_coords = np.asarray(pixel_coords).copy()
        
        # Flatten to (N, 2) for processing
        if len(pixel_coords.shape) == 3:
            pixel_coords = pixel_coords.reshape(-1, 2)
        
        # Step 1: Undo the y-axis inversion
        if resize_factor:
            pixel_coords[:, 1] = viz_img_size[1] * 0.46 - 1 - pixel_coords[:, 1]
        else:
            pixel_coords[:, 1] = viz_img_size[1] - 1 - pixel_coords[:, 1]
        
        pixel_coords = pixel_coords.astype(np.float64)
        if resize_factor:
            pixel_coords[:, 0] /= 0.35
            pixel_coords[:, 1] /= 0.46
        
        # Step 3: Undo the horizontal flip
        pixel_coords[:, 0] = viz_img_size[0] - pixel_coords[:, 0]
        
        # Step 4: Unproject from 2D to 3D
        xy_coords = self.unproject_pixels(
            pixel_coords,
            camera_height,
            camera_x_offset,
            camera_matrix,
            dist_coeffs
        )
        
        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            xy_coords = xy_coords.reshape(original_shape[0], original_shape[1], 2)
        
        return xy_coords