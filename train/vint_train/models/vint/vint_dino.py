import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import LastTokenMultiLayerDecoder

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
        positional_encoding_type: str = "peg",
    ) -> None:
        super(ViNTWithDINOTokens, self).__init__(context_size, len_traj_pred, learn_angle)
        self.encoding_size = encoding_size
        self.image_size = image_size
        self.context_size = context_size

        if "dino" in obs_encoder:
            self.vision_encoder = AutoModel.from_pretrained(obs_encoder)
            self.patch_size = self.vision_encoder.config.patch_size
        else:
            raise NotImplementedError

        assert positional_encoding_type in ["peg", "sinusoidal", "rope"], "positional_encoding_type must be one of 'peg', 'sinusoidal', or 'rope'"
        
        self.vision_projector = VisionProjector(
            input_dim = self.vision_encoder.config.hidden_size,
            output_dim = self.encoding_size,
            disable_peg = (positional_encoding_type != "peg"),
        )

        self.token_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.encoding_size,
        )
        self.positional_encoding_type = positional_encoding_type
        self.temporal_embedding = nn.Parameter(torch.zeros((1, self.context_size+1, self.encoding_size, 1)))

        if positional_encoding_type == "rope":
            self.grid_h = self.image_size[0] // self.patch_size // 2
            self.grid_w = self.image_size[1] // self.patch_size // 2
            
            dim = self.encoding_size // 2
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            
            h = torch.arange(self.grid_h).float()
            w = torch.arange(self.grid_w).float()
            
            freqs_h = torch.einsum("i,j->ij", h, inv_freq)
            freqs_w = torch.einsum("i,j->ij", w, inv_freq)
            
            self.register_buffer("cos_h", torch.cat((freqs_h, freqs_h), dim=-1).cos())
            self.register_buffer("sin_h", torch.cat((freqs_h, freqs_h), dim=-1).sin())
            self.register_buffer("cos_w", torch.cat((freqs_w, freqs_w), dim=-1).cos())
            self.register_buffer("sin_w", torch.cat((freqs_w, freqs_w), dim=-1).sin())

        self.decoder = LastTokenMultiLayerDecoder(
            embed_dim=self.encoding_size,
            seq_len=(self.context_size+1)*(self.image_size[0]//self.patch_size//2)*(self.image_size[1]//self.patch_size//2)+1,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
            apply_positional_encoding=positional_encoding_type == "sinusoidal",
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor,
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

        if self.positional_encoding_type in ["peg", "rope"]:
            image_tokens = image_tokens + self.temporal_embedding
        if self.positional_encoding_type == "rope":
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

        pred_token_emb = self.token_embedding(torch.tensor([0]*batch_size).to(obs_img.device))[:,None, :]
        image_tokens = image_tokens.permute((0, 3, 1, 2)) 
        tokens = torch.cat([
            image_tokens.reshape(
                batch_size,
                -1,
                self.encoding_size
            ),  
            pred_token_emb
        ], dim=1)
        
        final_repr = self.decoder(tokens)

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