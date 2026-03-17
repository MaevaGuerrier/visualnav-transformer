import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import TransformerEncoder, RoFormer
from vint_train.models.vint.rope import RotaryPositionEmbedding2D

from transformers import AutoModel

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-3] // 2, :, :]
    x2 = x[..., x.shape[-3] // 2 :, :, :]
    return torch.cat((-x2, x1), dim=-3)

class VisionProjector(nn.Module):
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
        f_0 = self.pw2(self.gelu(self.pw1(x)))
        f_1 = self.pool(f_0)
        if not self.disable_peg:
            tokens = self.dw(f_1) + f_1
        else:
            tokens = f_1
        return tokens.reshape(tokens.shape[0], self.output_dim, -1)


class ViNTWithDINOTokens(BaseModel):
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        context_size: int = 5,
        len_traj_pred: int = 5,
        learn_angle: bool = True,
        obs_encoder: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        encoding_size: int = 512,
        mha_num_attention_heads: int = 2,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        output_layers: List[int] = [256, 128, 64, 32],
        positional_encoding_type: str = "peg",
        separate_tokens_and_heads: bool = False,
        take_action_history: bool = False,
        action_enc_layers: List[int] = [128, 256]
    ) -> None:
        super(ViNTWithDINOTokens, self).__init__(context_size, len_traj_pred, learn_angle)
        self.encoding_size = encoding_size
        self.image_size = image_size
        self.context_size = context_size
        self.separate_tokens_and_heads = separate_tokens_and_heads
        self.take_action_history = take_action_history

        if "dino" in obs_encoder:
            self.vision_encoder = AutoModel.from_pretrained(obs_encoder)
            self.patch_size = self.vision_encoder.config.patch_size
        else:
            raise NotImplementedError

        assert positional_encoding_type in ["peg", "sinusoidal", "rope", "temporal_only"],\
            f"Unsupported positional encoding type: {positional_encoding_type}"
        
        self.vision_projector = VisionProjector(
            input_dim = self.vision_encoder.config.hidden_size,
            output_dim = self.encoding_size,
            disable_peg = (positional_encoding_type != "peg"),
        )

        self.positional_encoding_type = positional_encoding_type
        
        # Token(s) for readout from the decoder
        if self.separate_tokens_and_heads:
            # Separate tokens for distance and action prediction
            self.dist_token_embedding = nn.Embedding(
                num_embeddings=1,
                embedding_dim=self.encoding_size,
            )
            self.action_token_embedding = nn.Embedding(
                num_embeddings=1,
                embedding_dim=self.encoding_size,
            )
        else:
            # Shared token for both predictions
            self.token_embedding = nn.Embedding(
                num_embeddings=1,
                embedding_dim=self.encoding_size,
            )
        self.temporal_embedding = nn.Parameter(torch.zeros((1, self.context_size+1, self.encoding_size, 1)))

        # Adjust seq_len based on number of readout tokens
        num_readout_tokens = 2 if self.separate_tokens_and_heads else 1
        image_tokens_len = (self.context_size+1)*(self.image_size[0]//self.patch_size//2)*(self.image_size[1]//self.patch_size//2)
        
        if positional_encoding_type == "rope":
            self.grid_h = self.image_size[0] // self.patch_size // 2
            self.grid_w = self.image_size[1] // self.patch_size // 2
            
            # Create RoPE instance
            self.rope = RotaryPositionEmbedding2D(frequency=100.0)
            
            # Precompute 2D position grid for all timesteps
            S = self.context_size + 1
            num_patches_per_image = self.grid_h * self.grid_w
            
            # Create position grid for one image: [(h, w) for all patches]
            positions = torch.stack([
                torch.tensor([h, w])
                for h in range(self.grid_h)
                for w in range(self.grid_w)
            ])  # [num_patches_per_image, 2]
            
            # Expand for all timesteps
            positions = positions.unsqueeze(0).expand(S, -1, -1)  # [S, num_patches_per_image, 2]
            self.register_buffer('rope_positions', positions)
            
            # Create RoFormer decoder
            self.decoder = RoFormer(
                embed_dim=self.encoding_size,
                nhead=mha_num_attention_heads,
                num_layers=mha_num_attention_layers,
                ff_dim_factor=mha_ff_dim_factor,
                rope=self.rope,
            )
        else:
            self.rope = None
            self.decoder = TransformerEncoder(
                embed_dim=self.encoding_size,
                seq_len=image_tokens_len + num_readout_tokens,
                nhead=mha_num_attention_heads,
                num_layers=mha_num_attention_layers,
                ff_dim_factor=mha_ff_dim_factor,
                apply_positional_encoding=positional_encoding_type == "sinusoidal",
            )
        # Output layers for processing extracted tokens
        if self.separate_tokens_and_heads:
            # Separate output layers for distance and action
            dist_layers = [nn.Linear(self.encoding_size, output_layers[0]), nn.ReLU()]
            for i in range(len(output_layers) - 1):
                dist_layers.extend([nn.Linear(output_layers[i], output_layers[i + 1]), nn.ReLU()])
            self.dist_output_layers = nn.Sequential(*dist_layers)
            
            action_layers = [nn.Linear(self.encoding_size, output_layers[0]), nn.ReLU()]
            for i in range(len(output_layers) - 1):
                action_layers.extend([nn.Linear(output_layers[i], output_layers[i + 1]), nn.ReLU()])
            self.action_output_layers = nn.Sequential(*action_layers)
        else:
            # Shared output layers
            layers = [nn.Linear(self.encoding_size, output_layers[0]), nn.ReLU()]
            for i in range(len(output_layers) - 1):
                layers.extend([nn.Linear(output_layers[i], output_layers[i + 1]), nn.ReLU()])
            self.output_layers = nn.Sequential(*layers)
        
        self.dist_predictor = nn.Sequential(
            nn.Linear(output_layers[-1], 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(output_layers[-1], self.len_trajectory_pred * self.num_action_params),
        )

        action_enc_layers += [self.encoding_size]  # Ensure final layer matches decoder embedding size
        if self.take_action_history:
            self.action_history_layers = [nn.Linear(4 if self.learn_angle else 2, action_enc_layers[0]), nn.ReLU()]
            for i in range(len(action_enc_layers) - 1):
                self.action_history_layers.extend([nn.Linear(action_enc_layers[i], action_enc_layers[i + 1]), nn.ReLU()])
            self.action_history_encoder = nn.Sequential(*self.action_history_layers)

    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor, action_history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = obs_img.shape[0]
        obs_img_split = torch.split(obs_img, 3, dim=1)[1:]

        # (0th_batch_obs_1, 0th_batch_obs_2, ..., 0th_batch_obs_context_size,
        #  1st_batch_obs_1, ..., batch_size-1_batch_obs_context_size)
        img = torch.concat((*obs_img_split, goal_img), dim=0)

        vision_outputs = self.vision_encoder(pixel_values=img)
        vision_outputs = vision_outputs.last_hidden_state[:, 5:, :].reshape(
            batch_size * (self.context_size + 1),
            self.image_size[0] // self.patch_size,
            self.image_size[1] // self.patch_size,
            self.vision_encoder.config.hidden_size,
        ) 
        vision_outputs = vision_outputs.permute(0, 3, 1, 2) 
        vision_encodings = self.vision_projector(vision_outputs)

        obs_tokens_flat = vision_encodings[:batch_size*self.context_size, :, :]
        obs_tokens = obs_tokens_flat.view(
            self.context_size,
            batch_size,
            self.encoding_size,
            self.image_size[0]//self.patch_size//2 * self.image_size[1]//self.patch_size//2,
        ).permute(1, 0, 2, 3)

        goal_tokens = vision_encodings[batch_size*self.context_size:, None, :, :]
        image_tokens = torch.cat([obs_tokens, goal_tokens], dim=1) 

        action_avail = self.take_action_history and action_history is not None
        if action_avail:
            # remove last action from history for encoding, as it corresponds to the current step
            action_history_emb = self.action_history_encoder(action_history[:, :-1])  # [B, context_size, action_enc_layers[-1]]

        if self.positional_encoding_type in ["peg", "rope", "temporal_only"]:
            image_tokens = image_tokens + self.temporal_embedding
            if action_avail:
                action_history_emb = action_history_emb + self.temporal_embedding[:, :self.context_size-1, :, 0]
        
        # Apply manual RoPE for non-RoFormer case (deprecated, kept for compatibility)
        if self.positional_encoding_type == "rope" and not hasattr(self, 'rope'):
            image_tokens = image_tokens.reshape(batch_size, self.context_size + 1, self.encoding_size, self.grid_h, self.grid_w)
            
            img_h = image_tokens[:, :, :self.encoding_size//2, :, :]
            img_w = image_tokens[:, :, self.encoding_size//2:, :, :]
            
            c_h = self.cos_h.T.reshape(1, 1, -1, self.grid_h, 1)
            s_h = self.sin_h.T.reshape(1, 1, -1, self.grid_h, 1)
            
            img_h = (img_h * c_h) + (rotate_half(img_h) * s_h)
            
            c_w = self.cos_w.T.reshape(1, 1, -1, 1, self.grid_w)
            s_w = self.sin_w.T.reshape(1, 1, -1, 1, self.grid_w)
            
            img_w = (img_w * c_w) + (rotate_half(img_w) * s_w)
            
            image_tokens = torch.cat([img_h, img_w], dim=2)
            image_tokens = image_tokens.reshape(batch_size, self.context_size + 1, self.encoding_size, -1)

        image_tokens = image_tokens.permute((0, 3, 1, 2)) 
        image_tokens_flat = image_tokens.reshape(
            batch_size,
            -1,
            self.encoding_size
        )
        
        device = obs_img.device
        
        if self.separate_tokens_and_heads:
            # Get both readout tokens
            dist_token_emb = self.dist_token_embedding(torch.zeros(batch_size, dtype=torch.long, device=device))[:, None, :]
            action_token_emb = self.action_token_embedding(torch.zeros(batch_size, dtype=torch.long, device=device))[:, None, :]
            non_image_tokens = torch.cat([dist_token_emb, action_token_emb], dim=1)  # [B, 2, C]
        else:
            # Shared token for both predictions
            non_image_tokens = self.token_embedding(torch.zeros(batch_size, dtype=torch.long, device=device))[:, None, :]  # [B, 1, C]

        if action_avail:
            non_image_tokens = torch.cat([action_history_emb, non_image_tokens], dim=1)  # [B, num_readout_tokens + context_size - 1, C]

        
        
        # Run encoder with RoFormer or TransformerEncoder
        if self.positional_encoding_type == "rope" and self.rope is not None:
            # Use RoFormer: expand precomputed positions to batch
            positions = self.rope_positions.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, S, num_patches, 2]
            positions = positions.reshape(batch_size, -1, 2)  # [B, S * num_patches, 2]
            
            # Process through RoFormer
            _, readout_out = self.decoder(
                rope_tokens=image_tokens_flat,
                rope_positions=positions,
                other_tokens=non_image_tokens
            )
            
            if self.separate_tokens_and_heads:
                dist_repr = self.dist_output_layers(readout_out[:, -2, :])   # First readout token
                action_repr = self.action_output_layers(readout_out[:, 1, :])  # Second readout token
            else:
                final_repr = self.output_layers(readout_out[:, -1, :])
        else:
            # Use TransformerEncoder: concatenate all tokens
            tokens = torch.cat([image_tokens_flat, non_image_tokens], dim=1)
            encoded_tokens = self.decoder(tokens)
            
            if self.separate_tokens_and_heads:
                dist_repr = self.dist_output_layers(encoded_tokens[:, -2, :])   # Second to last token
                action_repr = self.action_output_layers(encoded_tokens[:, -1, :])  # Last token
            else:
                final_repr = self.output_layers(encoded_tokens[:, -1, :])  # Last token
        
        if self.separate_tokens_and_heads:
            dist_pred = self.dist_predictor(dist_repr)
            action_pred = self.action_predictor(action_repr)
        else:
            dist_pred = self.dist_predictor(final_repr)
            action_pred = self.action_predictor(final_repr)

        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  
        return dist_pred, action_pred