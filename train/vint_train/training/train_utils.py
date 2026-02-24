import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points, compare_waypoints_pred_to_label
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy, numpy_to_img, RED, GREEN, CYAN, MAGENTA
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from timm.scheduler import CosineLRScheduler

# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

# Train utils for ViNT and GNM
def _compute_losses(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    action_mask: torch.Tensor = None,
    return_per_sample: bool = False,
):
    """
    Compute losses for distance and action prediction.

    Args:
        return_per_sample: if True, also return per-sample total losses
    """
    dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    def action_reduce_per_sample(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask) / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[:, :, :2], action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(action_pred[:, :, :2], start_dim=1),
        torch.flatten(action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "dist_loss": dist_loss,
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        action_orien_cos_sim = action_reduce(F.cosine_similarity(
            action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
        ))
        multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, 2:], start_dim=1),
            torch.flatten(action_label[:, :, 2:], start_dim=1),
            dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    results["total_loss"] = total_loss

    if return_per_sample:
        # Compute per-sample losses for high-loss detection
        dist_loss_per_sample = F.mse_loss(dist_pred.squeeze(-1), dist_label.float(), reduction='none')
        action_loss_per_sample = action_reduce_per_sample(F.mse_loss(action_pred, action_label, reduction="none"))
        total_loss_per_sample = alpha * 1e-2 * dist_loss_per_sample + (1 - alpha) * action_loss_per_sample
        results["total_loss_per_sample"] = total_loss_per_sample

    return results


def _log_data(
    i,
    epoch,
    num_batches,
    normalized,
    project_folder,
    num_images_log,
    loggers,
    obs_image,
    goal_image,
    action_pred,
    action_label,
    dist_pred,
    dist_label,
    goal_pos,
    dataset_index,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if print_log_freq != 0 and i % print_log_freq == 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if print_log_freq != 0 and i % print_log_freq == 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        visualize_dist_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            mode,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )
        visualize_traj_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dataset_index),
            to_numpy(goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            mode,
            normalized,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )


def _log_high_loss_samples(
    obs_images: torch.Tensor,
    goal_images: torch.Tensor,
    action_pred: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    dist_label: torch.Tensor,
    total_loss_per_sample: torch.Tensor,
    high_loss_indices: torch.Tensor,
    dataset_index: torch.Tensor,
    epoch: int,
    batch_idx: int,
    project_folder: str,
    normalized: bool,
    mode: str = "train",
    max_samples: int = 10,
    use_wandb: bool = True,
):
    """
    Log samples with unusually high loss for debugging.
    Uses existing visualization utilities from vint_train.visualizing.
    """
    num_to_log = min(len(high_loss_indices), max_samples)
    log_dir = os.path.join(project_folder, f"high_loss_{mode}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Convert tensors to numpy for visualization
    obs_images_np = to_numpy(obs_images)
    goal_images_np = to_numpy(goal_images)
    action_pred_np = to_numpy(action_pred)
    action_label_np = to_numpy(action_label)
    dataset_index_np = to_numpy(dataset_index)
    
    # Denormalize images (undo ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if obs_images_np.shape[1] > 3:
        # If there are multiple observation images concatenated, only denormalize the last 3 channels (the most recent observation)
        obs_images_np = obs_images_np[:, -3:, :, :]
    obs_images_np = obs_images_np * std[None, :, None, None] + mean[None, :, None, None]
    goal_images_np = goal_images_np * std[None, :, None, None] + mean[None, :, None, None]
    obs_images_np = np.clip(obs_images_np, 0, 1)
    goal_images_np = np.clip(goal_images_np, 0, 1)
    
    wandb_list = []
    
    for idx_in_batch, sample_idx in enumerate(high_loss_indices[:num_to_log]):
        sample_idx = sample_idx.item()
        loss_value = total_loss_per_sample[sample_idx].item()
        
        # Get data for this sample
        obs_img = obs_images_np[sample_idx]
        goal_img = goal_images_np[sample_idx]
        pred_waypoints = action_pred_np[sample_idx]
        label_waypoints = action_label_np[sample_idx]
        dataset_idx = int(dataset_index_np[sample_idx])
        
        # Get dataset name from data_config (already loaded at module level)
        dataset_names = sorted(list(data_config.keys()))
        dataset_name = dataset_names[dataset_idx] if dataset_idx < len(dataset_names) else "unknown"
        
        # Use existing visualization function
        save_path = os.path.join(log_dir, f"epoch{epoch}_batch{batch_idx}_sample{sample_idx}_loss{loss_value:.4f}.png")
        
        compare_waypoints_pred_to_label(
            obs_img=numpy_to_img(obs_img),
            goal_img=numpy_to_img(goal_img),
            dataset_name=dataset_name,
            goal_pos=np.array([0, 0]),  # Placeholder, actual goal pos not directly available
            pred_waypoints=pred_waypoints,
            label_waypoints=label_waypoints,
            save_path=save_path,
            display=False,
        )
        
        if use_wandb:
            wandb_list.append(wandb.Image(save_path, caption=f"Loss: {loss_value:.4f}"))
    
    # Log all high-loss samples to wandb at once
    if use_wandb and wandb_list:
        wandb.log({
            f"{mode}/high_loss_samples": wandb_list,
            f"{mode}/high_loss_count": len(wandb_list),
            f"{mode}/high_loss_threshold": total_loss_per_sample[high_loss_indices[0]].item(),
        }, commit=False)

def train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    scheduler: Optional[CosineLRScheduler] = None,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
    log_high_loss_samples: bool = False,
    high_loss_threshold: float = 10.0,
    max_high_loss_samples: int = 10,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
        log_high_loss_samples: whether to log samples with high loss
        high_loss_threshold: threshold for considering a loss as high
        max_high_loss_samples: maximum number of high loss samples to log
    """
    model.train()
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
        
        goal_image = transform(goal_image).to(device)
        model_outputs = model(obs_image, goal_image)

        dist_label = dist_label.to(device)
        action_label = action_label.to(device)
        action_mask = action_mask.to(device)

        optimizer.zero_grad()
      
        dist_pred, action_pred = model_outputs

        losses = _compute_losses(
            dist_label=dist_label,
            action_label=action_label,
            dist_pred=dist_pred,
            action_pred=action_pred,
            alpha=alpha,
            learn_angle=learn_angle,
            action_mask=action_mask,
            return_per_sample=log_high_loss_samples,
        )

        # Log high-loss samples if enabled
        if log_high_loss_samples and "total_loss_per_sample" in losses:
            total_loss_per_sample = losses["total_loss_per_sample"]
            high_loss_mask = total_loss_per_sample > high_loss_threshold
            if high_loss_mask.any():
                high_loss_indices = torch.where(high_loss_mask)[0]
                print(f"[Epoch {epoch} Batch {i}] Found {len(high_loss_indices)} high-loss samples (threshold: {high_loss_threshold})")
                _log_high_loss_samples(
                    obs_images=obs_image,
                    goal_images=goal_image,
                    action_pred=action_pred,
                    action_label=action_label,
                    dist_pred=dist_pred,
                    dist_label=dist_label,
                    total_loss_per_sample=total_loss_per_sample,
                    high_loss_indices=high_loss_indices,
                    dataset_index=dataset_index,
                    epoch=epoch,
                    batch_idx=i,
                    project_folder=project_folder,
                    normalized=normalized,
                    mode="train",
                    max_samples=max_high_loss_samples,
                    use_wandb=use_wandb,
                )

        losses["total_loss"].backward()
        optimizer.step()

        if scheduler is not None and isinstance(scheduler, CosineLRScheduler):
            scheduler.step(epoch + i / num_batches)

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        _log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=normalized,
            project_folder=project_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            goal_image=viz_goal_image,
            action_pred=action_pred,
            action_label=action_label,
            dist_pred=dist_pred,
            dist_label=dist_label,
            goal_pos=goal_pos,
            dataset_index=dataset_index,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
        )


def evaluate(
    eval_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,

):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
        eval_fraction (float): fraction of data to use for evaluation
        use_tqdm (bool): whether to use tqdm for logging
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
    multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", eval_type)
    total_loss_logger = Logger("total_loss", eval_type)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger("action_orien_cos_sim", eval_type)
        multi_action_orien_cos_sim_logger = Logger("multi_action_orien_cos_sim", eval_type)
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    viz_obs_image = None
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

            goal_image = transform(goal_image).to(device)
            model_outputs = model(obs_image, goal_image)

            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)

            dist_pred, action_pred = model_outputs

            losses = _compute_losses(
                dist_label=dist_label,
                action_label=action_label,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=alpha,
                learn_angle=learn_angle,
                action_mask=action_mask,
            )

            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())

    # Log data to wandb/console, with visualizations selected from the last batch
    _log_data(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        normalized=normalized,
        project_folder=project_folder,
        num_images_log=num_images_log,
        loggers=loggers,
        obs_image=viz_obs_image,
        goal_image=viz_goal_image,
        action_pred=action_pred,
        action_label=action_label,
        goal_pos=goal_pos,
        dist_pred=dist_pred,
        dist_label=dist_label,
        dataset_index=dataset_index,
        use_wandb=use_wandb,
        mode=eval_type,
        use_latest=False,
        wandb_increment_step=False,
    )

    return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()


# Train utils for NOMAD

def _compute_losses_nomad(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
):
    """
    Compute losses for distance and action prediction.
    """

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
    )
    uc_actions = model_output_dict['uc_actions']
    gc_actions = model_output_dict['gc_actions']
    gc_distance = model_output_dict['gc_distance']

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_action_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_action_label, reduction="none"))

    uc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    uc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(uc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    gc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    gc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(gc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results


def _log_high_loss_samples_nomad(
    obs_images: torch.Tensor,
    goal_images: torch.Tensor,
    actions: torch.Tensor,
    distance: torch.Tensor,
    dist_pred: torch.Tensor,
    diffusion_loss_per_sample: torch.Tensor,
    high_loss_indices: torch.Tensor,
    dataset_idx: torch.Tensor,
    epoch: int,
    batch_idx: int,
    project_folder: str,
    mode: str = "train",
    max_samples: int = 10,
    use_wandb: bool = True,
):
    """
    Log NoMaD samples with unusually high loss for debugging.
    Uses existing visualization utilities from vint_train.visualizing.
    """
    num_to_log = min(len(high_loss_indices), max_samples)
    log_dir = os.path.join(project_folder, f"high_loss_{mode}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Convert tensors to numpy for visualization
    obs_images_np = to_numpy(obs_images)
    goal_images_np = to_numpy(goal_images)
    actions_np = to_numpy(actions)
    
    # Denormalize images (undo ImageNet normalization) - take last 3 channels for obs
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # For obs_images, take the last 3 channels (most recent observation)
    if obs_images_np.shape[1] > 3:
        obs_images_np = obs_images_np[:, -3:, :, :]
    obs_images_np = obs_images_np * std[None, :, None, None] + mean[None, :, None, None]
    goal_images_np = goal_images_np * std[None, :, None, None] + mean[None, :, None, None]
    obs_images_np = np.clip(obs_images_np, 0, 1)
    goal_images_np = np.clip(goal_images_np, 0, 1)
    
    wandb_list = []
    
    for idx_in_batch, sample_idx in enumerate(high_loss_indices[:num_to_log]):
        sample_idx = sample_idx.item()
        loss_value = diffusion_loss_per_sample[sample_idx].item()
        
        # Get images and actions for this sample
        obs_img = obs_images_np[sample_idx]
        goal_img = goal_images_np[sample_idx]
        action_traj = actions_np[sample_idx]
        
        # Create figure with subplots using matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(18.5, 10.5))
        
        # Plot action trajectory in first subplot using existing utility
        start_pos = np.array([0, 0])
        plot_trajs_and_points(
            axes[0],
            [action_traj],
            [start_pos],
            traj_colors=[CYAN],
            point_colors=[GREEN],
            traj_labels=["actions"],
            point_labels=["robot"],
        )
        axes[0].set_title(f"Action Trajectory (Loss: {loss_value:.4f})")
        
        # Show observation image using numpy_to_img
        obs_pil = numpy_to_img(obs_img)
        axes[1].imshow(obs_pil)
        axes[1].set_title("Observation")
        axes[1].axis('off')
        
        # Show goal image using numpy_to_img
        goal_pil = numpy_to_img(goal_img)
        axes[2].imshow(goal_pil)
        axes[2].set_title("Goal")
        axes[2].axis('off')
        
        # Save figure
        save_path = os.path.join(log_dir, f"epoch{epoch}_batch{batch_idx}_sample{sample_idx}_loss{loss_value:.4f}.png")
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
        if use_wandb:
            wandb_list.append(wandb.Image(save_path, caption=f"Loss: {loss_value:.4f}"))
    
    # Log all high-loss samples to wandb at once
    if use_wandb and wandb_list:
        wandb.log({
            f"{mode}/high_loss_samples": wandb_list,
            f"{mode}/high_loss_count": len(wandb_list),
            f"{mode}/high_loss_threshold": diffusion_loss_per_sample[high_loss_indices[0]].item(),
        }, commit=False)

def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    scheduler: Optional[CosineLRScheduler] = None,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    log_high_loss_samples: bool = False,
    high_loss_threshold: float = 10.0,
    max_high_loss_samples: int = 10,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        log_high_loss_samples: whether to log samples with high loss
        high_loss_threshold: threshold for considering a loss as high
        max_high_loss_samples: maximum number of high loss samples to log
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask, 
            ) = data
            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
            
            # Get distance label
            distance = distance.float().to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)
            
            # Predict the noise residual
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

            def action_reduce_per_sample(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask) / (action_mask.mean() + 1e-2)

            # L2 loss
            diffusion_loss_full = F.mse_loss(noise_pred, noise, reduction="none")
            diffusion_loss = action_reduce(diffusion_loss_full)
            
            # Compute per-sample losses for high-loss detection
            if log_high_loss_samples:
                diffusion_loss_per_sample = action_reduce_per_sample(diffusion_loss_full)
                high_loss_mask = diffusion_loss_per_sample > high_loss_threshold
                if high_loss_mask.any():
                    high_loss_indices = torch.where(high_loss_mask)[0]
                    print(f"[Epoch {epoch} Batch {i}] Found {len(high_loss_indices)} high-loss samples (threshold: {high_loss_threshold})")
                    _log_high_loss_samples_nomad(
                        obs_images=batch_obs_images,
                        goal_images=batch_goal_images,
                        actions=actions,
                        distance=distance,
                        dist_pred=dist_pred,
                        diffusion_loss_per_sample=diffusion_loss_per_sample,
                        high_loss_indices=high_loss_indices,
                        dataset_idx=dataset_idx,
                        epoch=epoch,
                        batch_idx=i,
                        project_folder=project_folder,
                        mode="train",
                        max_samples=max_high_loss_samples,
                        use_wandb=use_wandb,
                    )
            
            # Total loss
            loss = alpha * dist_loss + (1-alpha) * diffusion_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None and isinstance(scheduler, CosineLRScheduler):
                scheduler.step(epoch + i / num_batches)

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"diffusion_loss": diffusion_loss.item()})


            if i % print_log_freq == 0:
                losses = _compute_losses_nomad(
                            ema_model.averaged_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )


def evaluate_nomad(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model.averaged_model
    ema_model.eval()
    
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
            ) = data
            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
            no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            rand_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=rand_goal_mask)

            obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
            obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

            goal_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)

            distance = distance.to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            ### RANDOM MASK ERROR ###
            # Predict the noise residual
            rand_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=rand_mask_cond)
            
            # L2 loss
            rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)
            
            ### NO MASK ERROR ###
            # Predict the noise residual
            no_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=obsgoal_cond)
            
            # L2 loss
            no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

            ### GOAL MASK ERROR ###
            # predict the noise residual
            goal_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=goal_mask_cond)
            
            # L2 loss
            goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)
            
            # Logging
            loss_cpu = rand_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            wandb.log({"diffusion_eval_loss (random masking)": rand_mask_loss})
            wandb.log({"diffusion_eval_loss (no masking)": no_mask_loss})
            wandb.log({"diffusion_eval_loss (goal masking)": goal_mask_loss})

            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_nomad(
                            ema_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )


# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
    # obs_cond = obs_cond.flatten(start_dim=1)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)  
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output


    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample

    uc_actions = get_action(diffusion_output, ACTION_STATS)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_actions = get_action(diffusion_output, ACTION_STATS)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
    }


def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    
    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        uc_actions_list.append(to_numpy(model_output_dict['uc_actions']))
        gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))
        gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate([
            uc_actions,
            gc_actions,
            action_label[None],
        ], axis=0)
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)


