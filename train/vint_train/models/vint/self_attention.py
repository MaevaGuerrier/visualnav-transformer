import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from vint_train.models.vint.rope import RotaryPositionEmbedding2D


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.shape[1], :]
        return x

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, pos=None, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=(
                    (attn_mask)[:, None].repeat(1, self.num_heads, 1, 1)
                    if attn_mask is not None
                    else None
                ),
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer using nn.MultiheadAttention.
    
    This is equivalent to nn.TransformerEncoderLayer but gives us full control
    over weight initialization since each layer is independently created.
    
    Uses Pre-LN architecture (norm_first=True) for better training stability.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu", norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block with Pre-LN
        if self.norm_first:
            # Pre-LN: norm -> attention -> residual
            src2 = self.norm1(src)
            src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            
            # FFN block
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            # Post-LN: attention -> residual -> norm
            src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src = self.norm1(src + self.dropout1(src2))
            
            # FFN block
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(src2))
        
        return src


class TransformerDecoderLayer(nn.Module):
    """
    Custom Transformer Decoder Layer using nn.MultiheadAttention.
    
    This is equivalent to nn.TransformerDecoderLayer but gives us full control
    over weight initialization since each layer is independently created.
    
    Uses Pre-LN architecture (norm_first=True) for better training stability.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu", norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Cross-attention (encoder-decoder attention)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention block with Pre-LN
        if self.norm_first:
            # Self-attention
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            
            # Cross-attention
            tgt2 = self.norm2(tgt)
            tgt2, _ = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
            tgt = tgt + self.dropout2(tgt2)
            
            # FFN block
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            # Post-LN (not typically used with norm_first=True)
            tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt2))
            
            tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
            tgt = self.norm2(tgt + self.dropout2(tgt2))
            
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = self.norm3(tgt + self.dropout3(tgt2))
        
        return tgt


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder that applies positional encoding and transformer layers.
    Returns the full sequence of token representations.
    The caller is responsible for extracting what they need (last token, flatten all, etc.)
    """
    def __init__(self, embed_dim=512, seq_len=6, nhead=8, num_layers=8, ff_dim_factor=4, apply_positional_encoding=True):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len) if apply_positional_encoding else None
        
        # Create independent encoder layers
        dim_feedforward = ff_dim_factor * embed_dim
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation="gelu",
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            x: [batch_size, seq_len, embed_dim] - full sequence of token representations
        """
        if self.positional_encoding:
            x = self.positional_encoding(x)
        
        # Apply each layer independently
        for layer in self.layers:
            x = layer(x)
        
        return x

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(TransformerEncoderDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        
        # Create independent decoder layers instead of using TransformerDecoder
        # This allows each layer to have its own initialized weights
        dim_feedforward = ff_dim_factor * embed_dim
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation="gelu",
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
        
        self.output_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, encoder_input, decoder_input):
        if self.positional_encoding:
            encoder_input = self.positional_encoding(encoder_input)
            decoder_input = self.positional_encoding(decoder_input)
        
        x = decoder_input
        # Apply each decoder layer independently with cross-attention to encoder_input
        for layer in self.layers:
            x = layer(x, encoder_input)
        
        # currently, x is [batch_size, seq_len, embed_dim]
        x = x[:, -1, :]  # take only the last token
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x


class RoFormerLayer(nn.Module):
    """
    Single layer of RoFormer: Pre-LN transformer layer using Attention class.
    
    Applies RoPE to a subset of tokens (rope_tokens) while other tokens
    get no position encoding. All tokens participate in self-attention.
    """
    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        rope: Optional[RotaryPositionEmbedding2D] = None,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.rope = rope
        
        # Self-attention using Attention class
        self.self_attn = Attention(
            dim=embed_dim,
            num_heads=nhead,
            qkv_bias=False,
            proj_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
            rope=rope,
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feedforward network
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(
        self,
        src: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [B, N, C] concatenated tokens (rope_tokens + other_tokens)
            pos: [B, N_rope, 2] 2D positions for rope tokens, or None
            attn_mask: Optional attention mask
        Returns:
            [B, N, C] output tokens
        """
        # Self-attention block with Pre-LN
        if self.norm_first:
            src2 = self.norm1(src)
            src2 = self.self_attn(src2, pos=pos, attn_mask=attn_mask)
            src = src + self.dropout1(src2)
            
            # FFN block
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            # Post-LN
            src2 = self.self_attn(src, pos=pos, attn_mask=attn_mask)
            src = self.norm1(src + self.dropout1(src2))
            
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(src2))
        
        return src

class RoFormer(nn.Module):
    """
    RoFormer: Transformer encoder that applies 2D RoPE to specific tokens.
    
    Separates tokens into two groups:
    - rope_tokens: tokens that receive 2D RoPE (e.g., image patch tokens)
    - other_tokens: tokens without position encoding (e.g., readout tokens)
    
    All tokens are concatenated and processed together through self-attention layers.
    """
    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        ff_dim_factor: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        rope: Optional[RotaryPositionEmbedding2D] = None,
    ) -> None:
        """
        Args:
            embed_dim: Dimension of token embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim_factor: Feedforward dimension = ff_dim_factor * embed_dim
            dropout: Dropout rate
            activation: Activation function ("gelu" or "relu")
            norm_first: Use Pre-LN architecture if True
            rope: RotaryPositionEmbedding2D instance, or None for no RoPE
        """
        super().__init__()
        self.rope = rope
        
        dim_feedforward = ff_dim_factor * embed_dim
        self.layers = nn.ModuleList([
            RoFormerLayer(
                embed_dim=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                rope=rope,
            ) for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(
        self,
        rope_tokens: Optional[torch.Tensor] = None,
        rope_positions: Optional[torch.Tensor] = None,
        other_tokens: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through RoFormer.
        
        Args:
            rope_tokens: [B, N_rope, C] tokens to apply RoPE to (e.g., image patches)
            rope_positions: [B, N_rope, 2] 2D positions for rope tokens
            other_tokens: [B, N_other, C] tokens without RoPE (e.g., readout tokens)
            attn_mask: Optional attention mask [B, N_total, N_total]
            
        Returns:
            Tuple of (rope_tokens_out, other_tokens_out) with same shapes as inputs.
            If an input was None, the corresponding output is None.
        """
        # Validate inputs
        if rope_tokens is None and other_tokens is None:
            raise ValueError("At least one of rope_tokens or other_tokens must be provided")
        
        if rope_tokens is not None and self.rope is not None and rope_positions is None:
            raise ValueError("rope_positions must be provided when rope_tokens is given and rope is enabled")
        
        # Concatenate tokens for self-attention
        tokens_list = []
        rope_len = 0
        
        if rope_tokens is not None:
            rope_len = rope_tokens.shape[1]
            tokens_list.append(rope_tokens)
        
        if other_tokens is not None:
            tokens_list.append(other_tokens)
        
        if len(tokens_list) == 1:
            x = tokens_list[0]
        else:
            x = torch.cat(tokens_list, dim=1)
        
        # Build position tensor for Attention class
        # The Attention class expects pos of shape [B, N, 2] where N is total sequence length
        # For non-rope tokens, we can pass None or dummy positions (they won't be used if rope is None)
        pos = None
        if self.rope is not None and rope_tokens is not None:
            if other_tokens is not None:
                # Create dummy positions for other_tokens (won't be used since rope only applies to q,k from rope portion)
                # But we need to match the concatenated sequence length
                B, N_other, _ = other_tokens.shape
                device = rope_positions.device
                dummy_pos = torch.zeros(B, N_other, 2, device=device, dtype=rope_positions.dtype)
                pos = torch.cat([rope_positions+1, dummy_pos], dim=1)
            else:
                pos = rope_positions
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, pos=pos, attn_mask=attn_mask)
        
        # Split outputs back
        if rope_tokens is not None and other_tokens is not None:
            rope_out = x[:, :rope_len, :]
            other_out = x[:, rope_len:, :]
            return rope_out, other_out
        elif rope_tokens is not None:
            return x, None
        else:
            return None, x
