import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import TransformerEncoder

from depth_anything_3.api import DepthAnything3
from einops import rearrange


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-3] // 2, :, :]
    x2 = x[..., x.shape[-3] // 2 :, :, :]
    return torch.cat((-x2, x1), dim=-3)


class VisionProjector(nn.Module):
    """
    Projects vision features to the desired encoding size with downsampling.
    Uses 1x1 convolutions followed by average pooling.
    """
    def __init__(self, input_dim: int, output_dim: int, pool_size: int = 2, disable_peg: bool = False) -> None:
        super(VisionProjector, self).__init__()
        self.output_dim = output_dim
        self.pw1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.pw2 = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)

        self.disable_peg = disable_peg
        if not self.disable_peg:
            self.dw = nn.Conv2d(
                output_dim,
                output_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=output_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            tokens: (B, output_dim, H*W//pool_size^2) tensor
        """
        f_0 = self.pw2(self.gelu(self.pw1(x)))
        f_1 = self.pool(f_0)
        if not self.disable_peg:
            tokens = self.dw(f_1) + f_1
        else:
            tokens = f_1
        return tokens.reshape(tokens.shape[0], self.output_dim, -1)


class ViNTWithDepthAnything(BaseModel):
    """
    ViNT model using Depth-Anything's pretrained DINOv2 as the vision encoder.
    
    Key features:
    - Processes all images (context + goal) through a single DINO forward pass
    - Removes CLS and register tokens, keeps only patch tokens
    - Applies downsampling and projection via VisionProjector
    - Adds learnable readout tokens for distance/action prediction
    - Optional temporal positional encoding before DINO (since DA's DINO is permutation-invariant)
    
    Args:
        image_size: Tuple of (width, height) for input images
        context_size: Number of past observations to use
        len_traj_pred: Number of future waypoints to predict
        learn_angle: Whether to predict robot yaw
        obs_encoder: Depth-Anything model name (e.g., "depth-anything/DA3METRIC-LARGE")
        encoding_size: Size of the encoded features
        mha_num_attention_heads: Number of attention heads in transformer
        mha_num_attention_layers: Number of transformer layers
        mha_ff_dim_factor: Feedforward dimension factor
        output_layers: List of hidden layer sizes for output MLP
        positional_encoding_type: Type of positional encoding ("peg", "sinusoidal", "rope", "temporal_before_dino")
        separate_tokens_and_heads: If True, use separate readout tokens for distance and action
        add_temporal_pe: If True, add temporal positional encoding before DINO
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        context_size: int = 5,
        len_traj_pred: int = 5,
        learn_angle: bool = True,
        obs_encoder: str = "depth-anything/DA3METRIC-LARGE",
        encoding_size: int = 512,
        mha_num_attention_heads: int = 2,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        output_layers: List[int] = [256, 128, 64, 32],
        positional_encoding_type: str = "peg",
        separate_tokens_and_heads: bool = False,
        add_temporal_pe: bool = False,
        add_temporal_pe_dino: bool = False,
    ) -> None:
        super(ViNTWithDepthAnything, self).__init__(context_size, len_traj_pred, learn_angle)
        self.encoding_size = encoding_size
        self.image_size = image_size
        self.context_size = context_size
        self.separate_tokens_and_heads = separate_tokens_and_heads
        self.add_temporal_pe = add_temporal_pe
        self.add_temporal_pe_dino = add_temporal_pe_dino
        # Load Depth-Anything model and extract DINOv2 backbone
        self.da_model = DepthAnything3.from_pretrained(obs_encoder)
        self.vision_encoder = self.da_model.model.backbone.pretrained
        self.vision_encoder.eval()
        
        # Get DINO config
        self.embed_dim = self.vision_encoder.embed_dim
        self.patch_size = self.vision_encoder.patch_size
        self.num_register_tokens = self.vision_encoder.num_register_tokens
        
        # Calculate spatial dimensions after patch embedding
        self.grid_h = self.image_size[1] // self.patch_size
        self.grid_w = self.image_size[0] // self.patch_size
        self.num_patches = self.grid_h * self.grid_w
        
        # Number of tokens to remove (CLS + registers)
        self.num_special_tokens = 1 + self.num_register_tokens
        
        # Vision projector: projects from DINO dim to encoding_size with downsampling
        assert positional_encoding_type in ["peg", "sinusoidal", "rope"], \
            "positional_encoding_type must be one of 'peg', 'sinusoidal', 'rope'"
        
        self.positional_encoding_type = positional_encoding_type
        
        self.vision_projector = VisionProjector(
            input_dim=self.embed_dim,
            output_dim=self.encoding_size,
            pool_size=2,
            disable_peg=(positional_encoding_type not in ["peg"]),
        )
        
        # Temporal positional encoding (added before DINO if add_temporal_pe is True)
        if self.add_temporal_pe_dino:
            # Learnable temporal embeddings for each timestep
            self.temporal_pe_dino = nn.Parameter(
                torch.zeros(1, self.context_size + 1, self.embed_dim)
            )
            nn.init.normal_(self.temporal_pe_dino, std=0.02)
        
        # Readout tokens for aggregating information
        num_readout_tokens = 2 if self.separate_tokens_and_heads else 1
        self.num_readout_tokens = num_readout_tokens
        self.readout_tokens = nn.Parameter(
            torch.zeros(1, num_readout_tokens, self.encoding_size)
        )
        nn.init.normal_(self.readout_tokens, std=0.02)
        
        # Spatial positional encoding for image tokens
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, self.context_size + 1, self.encoding_size, 1)
        )
        nn.init.normal_(self.temporal_embedding, std=0.02)
        
        # RoPE positional encoding (if using rope)
        if positional_encoding_type == "rope":
            self.proj_grid_h = self.grid_h // 2  # After pooling in VisionProjector
            self.proj_grid_w = self.grid_w // 2
            
            dim = self.encoding_size // 2
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            
            h = torch.arange(self.proj_grid_h).float()
            w = torch.arange(self.proj_grid_w).float()
            
            freqs_h = torch.einsum("i,j->ij", h, inv_freq)
            freqs_w = torch.einsum("i,j->ij", w, inv_freq)
            
            self.register_buffer("cos_h", torch.cat((freqs_h, freqs_h), dim=-1).cos())
            self.register_buffer("sin_h", torch.cat((freqs_h, freqs_h), dim=-1).sin())
            self.register_buffer("cos_w", torch.cat((freqs_w, freqs_w), dim=-1).cos())
            self.register_buffer("sin_w", torch.cat((freqs_w, freqs_w), dim=-1).sin())
        
        # Transformer decoder
        # Input: all image tokens from all timesteps + readout tokens
        num_spatial_tokens_per_image = (self.grid_h // 2) * (self.grid_w // 2)  # After pooling
        total_image_tokens = (self.context_size + 1) * num_spatial_tokens_per_image
        seq_len = total_image_tokens + num_readout_tokens
        
        self.decoder = TransformerEncoder(
            embed_dim=self.encoding_size,
            seq_len=seq_len,
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
            apply_positional_encoding=(positional_encoding_type == "sinusoidal"),
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.encoding_size, output_layers[0]),
            nn.ReLU(),
        )
        for i in range(len(output_layers) - 1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i + 1]))
            self.output_layers.append(nn.ReLU())
        
        # Distance and action predictors
        self.dist_predictor = nn.Linear(output_layers[-1], 1)
        self.action_predictor = nn.Linear(
            output_layers[-1], self.len_trajectory_pred * self.num_action_params
        )
    
    def extract_dino_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features from DINOv2 backbone.
        
        Args:
            x: Input images (B*S, 3, H, W)
        Returns:
            patch_features: (B*S, num_patches, embed_dim) without CLS/reg tokens
        """
        B_S = x.shape[0]
        
        # Get patch embeddings
        patches = self.vision_encoder.patch_embed(x)  # (B*S, num_patches, embed_dim)
        
        # Prepare CLS token
        cls_tokens = self.vision_encoder.cls_token.expand(B_S, -1, -1)  # (B*S, 1, embed_dim)
        
        # Concatenate CLS + patches
        tokens = torch.cat([cls_tokens, patches], dim=1)  # (B*S, 1 + num_patches, embed_dim)
        
        # Add positional embeddings
        H, W = x.shape[-2:]
        tokens = tokens + self.vision_encoder.interpolate_pos_encoding(tokens, W, H)
        
        # Add register tokens if present
        if self.vision_encoder.register_tokens is not None:
            register_tokens = self.vision_encoder.register_tokens.expand(B_S, -1, -1)
            tokens = torch.cat([
                tokens[:, :1],  # CLS
                register_tokens,  # REGs
                tokens[:, 1:]   # Patches
            ], dim=1)

        if self.add_temporal_pe_dino:
            # Add temporal positional encoding to all tokens (including CLS/reg) before DINO blocks
            temporal_pe_expanded = self.temporal_pe_dino.expand(B_S, -1, -1)  # (B*S, S+1, embed_dim)
            temporal_pe_expanded = temporal_pe_expanded.reshape(B_S, 1, self.embed_dim)  # Broadcast to all tokens
            tokens = tokens + temporal_pe_expanded
        
        # Pass through transformer blocks
        for blk in self.vision_encoder.blocks:
            tokens = blk(tokens)
        
        # Apply final norm
        tokens = self.vision_encoder.norm(tokens)
        
        # Remove CLS and register tokens, keep only patch tokens
        patch_features = tokens[:, self.num_special_tokens:, :]  # (B*S, num_patches, embed_dim)
        
        return patch_features
    
    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            obs_img: (B, 3*(context_size+1), H, W) - stacked observation images
            goal_img: (B, 3, H, W) - goal image
        Returns:
            dist_pred: (B, 1) - predicted distance to goal
            action_pred: (B, len_traj_pred, num_action_params) - predicted waypoints
        """
        batch_size = obs_img.shape[0]
        
        # Split obs_img into individual observations
        # obs_img has shape (B, 3*(context_size+1), H, W)
        # We need to split into (context_size+1) images of 3 channels each
        obs_img_split = torch.split(obs_img, 3, dim=1)[1:]  # List of (B, 3, H, W) tensors
        
        # The first element might be empty or padding depending on data format
        # Typically obs_img contains context observations, and we concatenate goal
        # Based on vint_dino.py, obs_img includes context observations
        # Let's take the last context_size+1 images
        obs_images = list(obs_img_split[-(self.context_size + 1):])
        
        # Concatenate all images: context observations + goal
        all_images = torch.cat([*obs_images, goal_img], dim=0)  # (B*(S+1), 3, H, W)
        
        # Reshape for temporal PE addition if needed
        S = self.context_size + 1
        all_images_batch = all_images.view(batch_size, S, 3, self.image_size[1], self.image_size[0])
        
        # Flatten for DINO processing
        all_images_flat = all_images_batch.view(batch_size * S, 3, self.image_size[1], self.image_size[0])
        
        # Extract DINO features (without CLS/reg tokens)
        with torch.no_grad() if not self.training else torch.enable_grad():
            dino_features = self.extract_dino_features(all_images_flat)  # (B*S, num_patches, embed_dim)
        
        # Add temporal PE if enabled (by adding to all patches of each timestep)
        if self.add_temporal_pe:
            temporal_pe_expanded = self.temporal_pe.view(1, S, 1, self.embed_dim).expand(batch_size, -1, self.num_patches, -1)
            temporal_pe_expanded = temporal_pe_expanded.reshape(batch_size * S, self.num_patches, self.embed_dim)
            dino_features = dino_features + temporal_pe_expanded
        
        # Reshape to spatial format for projection: (B*S, embed_dim, grid_h, grid_w)
        dino_features = dino_features.permute(0, 2, 1)  # (B*S, embed_dim, num_patches)
        dino_features = dino_features.view(batch_size * S, self.embed_dim, self.grid_h, self.grid_w)
        
        # Apply vision projector (downsample + projection)
        projected_features = self.vision_projector(dino_features)  # (B*S, encoding_size, num_tokens)
        
        # Reshape to (B, S, encoding_size, num_spatial_tokens)
        num_spatial_tokens = projected_features.shape[-1]
        projected_features = projected_features.view(batch_size, S, self.encoding_size, num_spatial_tokens)
        
        # Add spatial-temporal positional encoding
        if self.positional_encoding_type in ["peg", "rope"]:
            projected_features = projected_features + self.temporal_embedding
        
        # Apply RoPE if enabled
        if self.positional_encoding_type == "rope":
            projected_features = projected_features.view(batch_size, S, self.encoding_size, self.proj_grid_h, self.proj_grid_w)
            
            img_h = projected_features[:, :, :self.encoding_size // 2, :, :]
            img_w = projected_features[:, :, self.encoding_size // 2:, :, :]
            
            c_h = self.cos_h.T.reshape(1, 1, -1, self.proj_grid_h, 1)
            s_h = self.sin_h.T.reshape(1, 1, -1, self.proj_grid_h, 1)
            img_h = (img_h * c_h) + (rotate_half(img_h) * s_h)
            
            c_w = self.cos_w.T.reshape(1, 1, -1, 1, self.proj_grid_w)
            s_w = self.sin_w.T.reshape(1, 1, -1, 1, self.proj_grid_w)
            img_w = (img_w * c_w) + (rotate_half(img_w) * s_w)
            
            projected_features = torch.cat([img_h, img_w], dim=2)
            projected_features = projected_features.view(batch_size, S, self.encoding_size, -1)
        
        # Flatten all image tokens: (B, S * num_spatial_tokens, encoding_size)
        image_tokens = projected_features.permute(0, 1, 3, 2)  # (B, S, num_spatial_tokens, encoding_size)
        image_tokens = image_tokens.reshape(batch_size, S * num_spatial_tokens, self.encoding_size)
        
        # Expand and add readout tokens
        readout_tokens = self.readout_tokens.expand(batch_size, -1, -1)  # (B, num_readout_tokens, encoding_size)
        all_tokens = torch.cat([image_tokens, readout_tokens], dim=1)  # (B, seq_len, encoding_size)
        
        # Pass through transformer decoder
        encoded_tokens = self.decoder(all_tokens)  # (B, seq_len, encoding_size)
        
        # Extract readout token outputs
        if self.separate_tokens_and_heads:
            # Separate tokens for distance and action
            dist_repr = self.output_layers(encoded_tokens[:, -2, :])  # Second to last token
            action_repr = self.output_layers(encoded_tokens[:, -1, :])  # Last token
            
            dist_pred = self.dist_predictor(dist_repr)
            action_pred = self.action_predictor(action_repr)
        else:
            # Shared token for both predictions
            final_repr = self.output_layers(encoded_tokens[:, -1, :])
            dist_pred = self.dist_predictor(final_repr)
            action_pred = self.action_predictor(final_repr)
        
        # Reshape action prediction
        action_pred = action_pred.reshape(
            batch_size, self.len_trajectory_pred, self.num_action_params
        )
        
        # Convert deltas to waypoints
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
        
        # Normalize angle if needed
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )
        
        return dist_pred, action_pred
