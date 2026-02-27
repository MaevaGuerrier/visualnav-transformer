import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
