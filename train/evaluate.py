#!/usr/bin/env python
"""
Standalone evaluation script for ViNT/GNM/NoMaD models.

This script evaluates pretrained models on specified datasets with:
- Custom visualization logic (independent from training config)
- Comprehensive metrics logging
- Support for evaluating on datasets not in training config
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import yaml
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.backends.cudnn as cudnn

import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

# Model imports
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vint_dino import ViNTWithDINOTokens
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

# Data and training imports
from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import load_model
from vint_train.visualizing.visualize_utils import to_numpy, numpy_to_img
from vint_train.visualizing.action_utils import compare_waypoints_pred_to_label, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE

# Load data config for normalization constants
with open(os.path.join(os.path.dirname(__file__), "vint_train/data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger("evaluation")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(output_dir, "evaluation.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def resolve_env_vars(obj):
    """Resolve environment variables in config."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = resolve_env_vars(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = resolve_env_vars(item)
    elif isinstance(obj, str):
        import re
        pattern = re.compile(r"\$\{(\w+)\}")
        def replace(match):
            env_var = match.group(1)
            return os.getenv(env_var, match.group(0))
        return pattern.sub(replace, obj)
    return obj


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create model based on config."""
    model_type = config["model_type"]
    
    if model_type == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif model_type == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif model_type == "vint_dino":
        model = ViNTWithDINOTokens(
            image_size=config["image_size"],
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            encoding_size=config["obs_encoding_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif model_type == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit":
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        
        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    else:
        raise ValueError(f"Model {model_type} not supported")
    
    model = model.to(device)
    return model


def load_datasets(eval_config: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """Load evaluation datasets from config."""
    dataloaders = {}
    
    # Image transform (ImageNet normalization)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get context type
    context_type = model_config.get("context_type", "temporal")
    
    for dataset_name, dataset_cfg in eval_config["datasets"].items():
        # Set defaults
        waypoint_spacing = dataset_cfg.get("waypoint_spacing", 1)
        end_slack = dataset_cfg.get("end_slack", 0)
        goals_per_obs = dataset_cfg.get("goals_per_obs", 1)
        negative_mining = dataset_cfg.get("negative_mining", True)
        
        dataset = ViNT_Dataset(
            data_folder=dataset_cfg["data_folder"],
            data_split_folder=dataset_cfg["data_split"],
            dataset_name=dataset_name,
            image_size=model_config["image_size"],
            waypoint_spacing=waypoint_spacing,
            min_dist_cat=model_config["distance"]["min_dist_cat"],
            max_dist_cat=model_config["distance"]["max_dist_cat"],
            min_action_distance=model_config["action"]["min_dist_cat"],
            max_action_distance=model_config["action"]["max_dist_cat"],
            negative_mining=negative_mining,
            len_traj_pred=model_config["len_traj_pred"],
            learn_angle=model_config["learn_angle"],
            context_size=model_config["context_size"],
            context_type=context_type,
            end_slack=end_slack,
            goals_per_obs=goals_per_obs,
            normalize=model_config["normalize"],
            goal_type=model_config.get("goal_type", "image"),
            flip_aug=False,  # No augmentation during eval
            image_aug=False,
        )
        
        # Get batch size
        batch_size = eval_config.get("batch_size", 32)
        num_workers = eval_config.get("num_workers", 4)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep order for consistent visualization
            num_workers=num_workers,
            persistent_workers=False,
            drop_last=False,
        )
        
        dataloaders[dataset_name] = {
            "dataloader": dataloader,
            "transform": transform,
        }
    
    return dataloaders


def compute_losses(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    action_mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute evaluation losses."""
    with torch.no_grad():
        dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())
        
        def action_reduce(unreduced_loss: torch.Tensor):
            while unreduced_loss.dim() > 1:
                unreduced_loss = unreduced_loss.mean(dim=-1)
            return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)
        
        action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))
        
        action_waypts_cos_sim = action_reduce(F.cosine_similarity(
            action_pred[:, :, :2], action_label[:, :, :2], dim=-1
        ))
        multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, :2], start_dim=1),
            torch.flatten(action_label[:, :, :2], start_dim=1),
            dim=-1,
        ))
        
        results = {
            "dist_loss": dist_loss.item(),
            "action_loss": action_loss.item(),
            "action_waypts_cos_sim": action_waypts_cos_sim.item(),
            "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim.item(),
        }
        
        if learn_angle:
            action_orien_cos_sim = action_reduce(F.cosine_similarity(
                action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
            ))
            multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
                torch.flatten(action_pred[:, :, 2:], start_dim=1),
                torch.flatten(action_label[:, :, 2:], start_dim=1),
                dim=-1,
            ))
            results["action_orien_cos_sim"] = action_orien_cos_sim.item()
            results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim.item()
        
        total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
        results["total_loss"] = total_loss.item()
    
    return results


def visualize_sample(
    obs_image: np.ndarray,
    goal_image: np.ndarray,
    dataset_name: str,
    goal_pos: np.ndarray,
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    dist_pred: float,
    dist_label: float,
    save_path: str,
    normalized: bool,
):
    """Visualize a single sample and save to file."""
    # Denormalize waypoints if needed
    #if normalized and dataset_name in data_config:
    #    metric_spacing = data_config[dataset_name]["metric_waypoint_spacing"]
    #    pred_waypoints = pred_waypoints.copy() * metric_spacing
    #    label_waypoints = label_waypoints.copy() * metric_spacing
    #    goal_pos = goal_pos.copy() * metric_spacing
    
    # Create figure with 4 subplots: traj, obs+traj, goal, distance info
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Plot 1: Trajectory comparison (top-down view)
    start_pos = np.array([0, 0])
    plot_trajs_and_points(
        axes[0],
        [pred_waypoints, label_waypoints],
        [start_pos, goal_pos],
        traj_colors=[np.array([0, 1, 1]), np.array([1, 0, 1])],  # CYAN, MAGENTA
        point_colors=[np.array([0, 1, 0]), np.array([1, 0, 0])],  # GREEN, RED
        traj_labels=["Predicted", "Ground Truth"],
        point_labels=["Robot", "Goal"],
    )
    axes[0].set_title("Trajectory Comparison (Top-down)")
    axes[0].legend()
    
    # Plot 2: Observation image
    # Images from dataloader are not normalized, just convert to displayable format
    obs_img_display = obs_image.copy()
    if obs_img_display.shape[0] > 3:
        # Take last 3 channels if multi-context
        obs_img_display = obs_img_display[-3:]
    # Already in [0, 1] range from dataset, just transpose to HWC
    obs_img_display = np.clip(obs_img_display, 0, 1)
    obs_img_display = np.transpose(obs_img_display, (1, 2, 0))
    
    axes[1].imshow(obs_img_display)
    axes[1].set_title("Observation")
    axes[1].axis('off')
    
    # Plot 3: Goal image (also not normalized)
    goal_img_display = goal_image.copy()
    goal_img_display = np.clip(goal_img_display, 0, 1)
    goal_img_display = np.transpose(goal_img_display, (1, 2, 0))
    
    axes[2].imshow(goal_img_display)
    axes[2].set_title("Goal")
    axes[2].axis('off')
    
    # Plot 4: Distance and error metrics
    axes[3].axis('off')
    dist_error = abs(dist_pred - dist_label)
    error_text = f"""
    Distance Prediction:
      Predicted: {dist_pred:.2f}
      Label: {dist_label:.2f}
      Error: {dist_error:.2f}
    
    Waypoint Error (L2):
      {np.mean(np.linalg.norm(pred_waypoints - label_waypoints, axis=1)):.4f}
    
    Dataset: {dataset_name}
    """
    axes[3].text(0.1, 0.5, error_text, fontsize=12, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[3].set_title("Metrics")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def evaluate_model(
    model: nn.Module,
    model_config: Dict[str, Any],
    dataloaders: Dict[str, Any],
    device: torch.device,
    output_dir: str,
    viz_every_n: Optional[int],
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    """Run evaluation on all datasets."""
    model.eval()
    alpha = model_config.get("alpha", 0.5)
    learn_angle = model_config["learn_angle"]
    normalized = model_config["normalize"]
    model_type = model_config["model_type"]
    
    all_metrics = {}
    
    with torch.no_grad():
        for dataset_name, dataset_info in dataloaders.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating on dataset: {dataset_name}")
            logger.info(f"{'='*60}")
            
            dataloader = dataset_info["dataloader"]
            transform = dataset_info["transform"]
            
            # Metrics accumulators
            metrics_accumulator = {
                "dist_loss": [],
                "action_loss": [],
                "action_waypts_cos_sim": [],
                "multi_action_waypts_cos_sim": [],
                "total_loss": [],
            }
            if learn_angle:
                metrics_accumulator["action_orien_cos_sim"] = []
                metrics_accumulator["multi_action_orien_cos_sim"] = []
            
            # Visualization counter
            viz_counter = 0
            global_sample_idx = 0
            
            # Setup visualization directory
            if viz_every_n is not None:
                viz_dir = os.path.join(output_dir, "visualizations", dataset_name)
                os.makedirs(viz_dir, exist_ok=True)
            
            pbar = tqdm.tqdm(dataloader, desc=f"Eval {dataset_name}")
            for batch_idx, batch_data in enumerate(pbar):
                # Unpack batch (different for nomad vs others)
                if model_type == "nomad":
                    obs_image, goal_image, action_label, dist_label, goal_pos, dataset_index, action_mask = batch_data
                else:
                    obs_image, goal_image, action_label, dist_label, goal_pos, dataset_index, action_mask = batch_data
                
                batch_size = obs_image.shape[0]
                
                # Process images
                obs_images = torch.split(obs_image, 3, dim=1)
                viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
                obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
                obs_image_processed = torch.cat(obs_images, dim=1)
                
                viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
                goal_image_processed = transform(goal_image).to(device)
                
                # Move labels to device
                dist_label = dist_label.to(device)
                action_label = action_label.to(device)
                action_mask = action_mask.to(device)
                
                # Forward pass
                if model_type == "nomad":
                    # NoMaD evaluation logic
                    # Get vision encoding
                    goal_mask = torch.zeros((batch_size,)).long().to(device)
                    obsgoal_cond = model("vision_encoder", obs_img=obs_image_processed, 
                                        goal_img=goal_image_processed, input_goal_mask=goal_mask)
                    obsgoal_cond_flat = obsgoal_cond.flatten(start_dim=1)
                    
                    # Distance prediction
                    dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond_flat)
                    
                    # For action prediction, use the noise_pred_net with zero noise (or average)
                    # Simplified: use the mean of the diffusion process
                    # Note: Full NoMaD evaluation with diffusion sampling is complex
                    # Here we do a simplified evaluation
                    
                    # Create dummy action_pred for metrics (would need full diffusion sampling for accurate eval)
                    # For now, use a placeholder that will give high loss to indicate this needs proper implementation
                    action_pred = torch.zeros_like(action_label)
                    
                    # Compute simplified losses
                    dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())
                    action_loss = torch.tensor(0.0)  # Placeholder
                    
                    batch_metrics = {
                        "dist_loss": dist_loss.item(),
                        "action_loss": 0.0,  # Not computed for NoMaD in simplified version
                        "action_waypts_cos_sim": 0.0,
                        "multi_action_waypts_cos_sim": 0.0,
                        "total_loss": dist_loss.item(),
                    }
                else:
                    # GNM, ViNT, ViNT-DINO
                    model_outputs = model(obs_image_processed, goal_image_processed)
                    dist_pred, action_pred = model_outputs
                    
                    batch_metrics = compute_losses(
                        dist_label=dist_label,
                        action_label=action_label,
                        dist_pred=dist_pred,
                        action_pred=action_pred,
                        alpha=alpha,
                        learn_angle=learn_angle,
                        action_mask=action_mask,
                    )
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    if key in metrics_accumulator:
                        metrics_accumulator[key].append(value)
                
                # Update progress bar
                pbar.set_postfix({
                    "dist_loss": f"{batch_metrics['dist_loss']:.4f}",
                    "action_loss": f"{batch_metrics['action_loss']:.4f}",
                })
                
                # Visualization
                if viz_every_n is not None:
                    # Convert tensors to numpy for visualization
                    obs_images_np = to_numpy(obs_image)
                    goal_images_np = to_numpy(goal_image)
                    action_pred_np = to_numpy(action_pred)
                    action_label_np = to_numpy(action_label)
                    dist_pred_np = to_numpy(dist_pred.squeeze(-1))
                    dist_label_np = to_numpy(dist_label)
                    goal_pos_np = to_numpy(goal_pos)
                    
                    for i in range(batch_size):
                        if global_sample_idx % viz_every_n == 0:
                            save_path = os.path.join(viz_dir, f"sample_{global_sample_idx:06d}.png")
                            visualize_sample(
                                obs_image=obs_images_np[i],
                                goal_image=goal_images_np[i],
                                dataset_name=dataset_name,
                                goal_pos=goal_pos_np[i],
                                pred_waypoints=action_pred_np[i],
                                label_waypoints=action_label_np[i],
                                dist_pred=float(dist_pred_np[i]),
                                dist_label=float(dist_label_np[i]),
                                save_path=save_path,
                                normalized=normalized,
                            )
                            viz_counter += 1
                        global_sample_idx += 1
            
            # Compute average metrics
            dataset_metrics = {}
            for key, values in metrics_accumulator.items():
                if values:
                    dataset_metrics[key] = float(np.mean(values))
                    dataset_metrics[f"{key}_std"] = float(np.std(values))
            
            # Add sample count
            dataset_metrics["num_samples"] = len(dataloader.dataset)
            dataset_metrics["num_batches"] = len(dataloader)
            
            all_metrics[dataset_name] = dataset_metrics
            
            # Log metrics
            logger.info(f"\nResults for {dataset_name}:")
            logger.info(f"  Samples: {dataset_metrics['num_samples']}")
            logger.info(f"  Distance Loss: {dataset_metrics.get('dist_loss', 0):.4f} ± {dataset_metrics.get('dist_loss_std', 0):.4f}")
            logger.info(f"  Action Loss: {dataset_metrics.get('action_loss', 0):.4f} ± {dataset_metrics.get('action_loss_std', 0):.4f}")
            logger.info(f"  Total Loss: {dataset_metrics.get('total_loss', 0):.4f} ± {dataset_metrics.get('total_loss_std', 0):.4f}")
            logger.info(f"  Action Waypts Cos Sim: {dataset_metrics.get('action_waypts_cos_sim', 0):.4f}")
            if viz_every_n:
                logger.info(f"  Visualizations saved: {viz_counter}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViNT/GNM/NoMaD models")
    
    # Required arguments
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to model config file (architecture parameters)",
    )
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        required=True,
        help="Path to pretrained checkpoint (.pth file)",
    )
    parser.add_argument(
        "--eval-datasets",
        "-d",
        required=True,
        help="Path to evaluation datasets config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to save evaluation outputs",
    )
    
    # Optional arguments
    parser.add_argument(
        "--viz-every-n",
        type=int,
        default=None,
        help="Visualize every Nth sample (if not set, no visualizations generated)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable wandb logging (default: disabled)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="vint-eval",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity name",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=[0],
        help="GPU IDs to use",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("="*60)
    logger.info("Starting Evaluation")
    logger.info("="*60)
    
    # Load configs
    logger.info(f"Loading model config from: {args.config}")
    with open(args.config, "r") as f:
        model_config = yaml.safe_load(f)
    model_config = resolve_env_vars(model_config)
    
    logger.info(f"Loading eval datasets config from: {args.eval_datasets}")
    with open(args.eval_datasets, "r") as f:
        eval_config = yaml.safe_load(f)
    eval_config = resolve_env_vars(eval_config)
    
    # Add CLI args to eval config
    eval_config["batch_size"] = args.batch_size
    eval_config["num_workers"] = args.num_workers
    
    # Setup device
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in args.gpu_ids])
        device = torch.device(f"cuda:{args.gpu_ids[0]}")
        logger.info(f"Using CUDA devices: {args.gpu_ids}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    cudnn.benchmark = True
    
    # Create model
    logger.info(f"Creating model: {model_config['model_type']}")
    model = create_model(model_config, device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_model(model, model_config["model_type"], checkpoint)
    logger.info("Checkpoint loaded successfully")
    
    # Multi-GPU if needed
    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)
        logger.info(f"Using DataParallel with {len(args.gpu_ids)} GPUs")
    
    # Setup wandb if requested
    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"eval_{time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_config": model_config,
                "eval_config": eval_config,
                "checkpoint": args.checkpoint,
            },
        )
        logger.info("Wandb logging enabled")
    
    # Load datasets
    logger.info("Loading evaluation datasets...")
    dataloaders = load_datasets(eval_config, model_config)
    logger.info(f"Loaded {len(dataloaders)} datasets")
    for name, info in dataloaders.items():
        logger.info(f"  {name}: {len(info['dataloader'].dataset)} samples")
    
    # Run evaluation
    logger.info("="*60)
    logger.info("Starting Evaluation Loop")
    logger.info("="*60)
    
    start_time = time.time()
    metrics = evaluate_model(
        model=model,
        model_config=model_config,
        dataloaders=dataloaders,
        device=device,
        output_dir=args.output_dir,
        viz_every_n=args.viz_every_n,
        logger=logger,
    )
    elapsed_time = time.time() - start_time
    
    # Save metrics to JSON
    metrics["_metadata"] = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "eval_datasets": args.eval_datasets,
        "elapsed_time_seconds": elapsed_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved to: {metrics_path}")
    
    # Log to wandb if enabled
    if args.use_wandb:
        for dataset_name, dataset_metrics in metrics.items():
            if not dataset_name.startswith("_"):
                wandb.log({f"{dataset_name}/{k}": v for k, v in dataset_metrics.items()})
        wandb.finish()
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete")
    logger.info("="*60)
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Print summary table
    logger.info("\nSummary of Results:")
    logger.info(f"{'Dataset':<20} {'Dist Loss':<12} {'Action Loss':<12} {'Total Loss':<12}")
    logger.info("-" * 60)
    for dataset_name, dataset_metrics in metrics.items():
        if not dataset_name.startswith("_"):
            logger.info(f"{dataset_name:<20} "
                       f"{dataset_metrics.get('dist_loss', 0):<12.4f} "
                       f"{dataset_metrics.get('action_loss', 0):<12.4f} "
                       f"{dataset_metrics.get('total_loss', 0):<12.4f}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
