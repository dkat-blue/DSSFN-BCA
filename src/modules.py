# src/modules.py
# Contains core building blocks for the DSSFN model:
# SelfAttention, CrossAttention, and PyramidalResidualBlock.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For sqrt in attention
import logging

class SelfAttention(nn.Module):
    """
    Self-attention Layer based on Section 3.3 and Figure 7 of the paper.
    Adapts for spectral (1D) or spatial (2D) input.
    """
    def __init__(self, in_dim):
        """
        Args:
            in_dim (int): Number of input channels.
        """
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        # Use a smaller intermediate dimension for query/key
        inter_dim = max(1, in_dim // 8)

        # --- Layers for Spatial Attention (4D input: B, C, H, W) ---
        self.query_conv_2d = nn.Conv2d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.key_conv_2d = nn.Conv2d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.value_conv_2d = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # --- Layers for Spectral Attention (3D input: B, C, L) ---
        # Using Q,K,V derived from input x, similar to spatial version but with Conv1d
        self.query_conv_1d = nn.Conv1d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.key_conv_1d = nn.Conv1d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.value_conv_1d = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1) # Softmax applied over the last dimension (N or L)

    def forward(self, x):
        """
        Forward pass for Self-Attention.

        Args:
            x (torch.Tensor): Input feature map.
                              Expected shape: (B, C, L) for spectral or (B, C, H, W) for spatial.

        Returns:
            torch.Tensor: Output feature map (same shape as input) with attention applied.
                          out = gamma * attention_output + x
        """
        if x.dim() == 3: # Spectral Attention (B, C, L) -> Attends over L dimension
            m_batchsize, C, length = x.size()
            proj_query = self.query_conv_1d(x).permute(0, 2, 1) # B, L, C'
            proj_key = self.key_conv_1d(x) # B, C', L
            energy = torch.bmm(proj_query, proj_key) # B, L, L (Attention map over length)

            attention = self.softmax(energy) # B, L, L
            proj_value = self.value_conv_1d(x) # B, C, L
            # Apply attention: B,L,L @ B,C,L -> needs proj_value as (B, L, C)
            proj_value_permuted = proj_value.permute(0, 2, 1) # B, L, C
            attn_output = torch.bmm(attention, proj_value_permuted) # B, L, L @ B, L, C -> B, L, C
            attn_output = attn_output.permute(0, 2, 1) # B, C, L (Back to original format)

            out = self.gamma * attn_output + x # Residual connection

        elif x.dim() == 4: # Spatial Attention (B, C, H, W) -> Attends over N=H*W dimension
            m_batchsize, C, height, width = x.size()
            N = height * width # Number of spatial locations (pixels)

            proj_query = self.query_conv_2d(x).view(m_batchsize, -1, N).permute(0, 2, 1) # B, N, C'
            proj_key = self.key_conv_2d(x).view(m_batchsize, -1, N) # B, C', N
            energy = torch.bmm(proj_query, proj_key) # B, N, N (Spatial attention map)
            attention = self.softmax(energy) # Softmax over spatial dimension N

            proj_value = self.value_conv_2d(x).view(m_batchsize, -1, N) # B, C, N

            # Apply spatial attention B,C,N @ B,N,N.T -> B,C,N
            attn_output = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, N
            attn_output = attn_output.view(m_batchsize, C, height, width) # Reshape back B, C, H, W

            out = self.gamma * attn_output + x # Residual connection

        else:
            raise ValueError("Input tensor must be 3D (B, C, L) or 4D (B, C, H, W)")

        return out

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention Layer.
    Allows one feature stream (query) to attend to another (context) using multiple heads.
    """
    def __init__(self, in_dim, num_heads=8, dropout=0.1):
        """
        Args:
            in_dim (int): Feature dimension of query and context streams. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(MultiHeadCrossAttention, self).__init__()
        if in_dim % num_heads != 0:
            raise ValueError(f"in_dim ({in_dim}) must be divisible by num_heads ({num_heads})")

        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.scale = self.head_dim ** -0.5 # Scaling factor for attention

        # Linear layers to project query, key, value for all heads at once
        self.to_q = nn.Linear(in_dim, in_dim, bias=False) # Projects query to Dim = num_heads * head_dim
        self.to_k = nn.Linear(in_dim, in_dim, bias=False) # Projects context to Dim
        self.to_v = nn.Linear(in_dim, in_dim, bias=False) # Projects context to Dim

        # Final linear layer after concatenating heads
        self.to_out = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Dropout(dropout)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context):
        """
        Forward pass for Multi-Head Cross-Attention.

        Args:
            x (torch.Tensor): Query feature map. Shape: (B, Seq_len_Q, Dim).
            context (torch.Tensor): Context feature map. Shape: (B, Seq_len_KV, Dim).

        Returns:
            torch.Tensor: Output feature map, shape: (B, Seq_len_Q, Dim).
        """
        B, N_Q, C = x.shape
        B, N_KV, C_ctx = context.shape
        if C != self.in_dim or C_ctx != self.in_dim:
             raise ValueError(f"Feature dimension mismatch: x({C}) or context({C_ctx}) != in_dim({self.in_dim})")

        # 1. Linear projections for Q, K, V
        q = self.to_q(x)  # (B, N_Q, Dim)
        k = self.to_k(context) # (B, N_KV, Dim)
        v = self.to_v(context) # (B, N_KV, Dim)

        # 2. Reshape and transpose for multi-head calculation
        # (B, Seq_len, Dim) -> (B, Seq_len, num_heads, head_dim) -> (B, num_heads, Seq_len, head_dim)
        q = q.view(B, N_Q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N_KV, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N_KV, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. Scaled dot-product attention per head
        # (B, H, N_Q, D_h) @ (B, H, D_h, N_KV) -> (B, H, N_Q, N_KV)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = self.softmax(attn_scores) # Softmax over N_KV dimension

        # 4. Apply attention to Value per head
        # (B, H, N_Q, N_KV) @ (B, H, N_KV, D_h) -> (B, H, N_Q, D_h)
        attn_output = torch.matmul(attn_probs, v)

        # 5. Concatenate heads and reshape back
        # (B, H, N_Q, D_h) -> (B, N_Q, H, D_h) -> (B, N_Q, Dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(B, N_Q, self.in_dim)

        # 6. Final linear layer and residual connection
        out = self.to_out(attn_output) + x # Add residual connection to the original query 'x'

        return out


class PyramidalResidualBlock(nn.Module):
    """
    Implements the Pyramidal Residual Block with Self-Attention integration.
    """
    def __init__(self, in_channels, out_channels, stride=1, is_1d=False):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first convolution (used for downsampling). Defaults to 1.
            is_1d (bool): True for spectral (1D conv), False for spatial (2D conv). Defaults to False.
        """
        super(PyramidalResidualBlock, self).__init__()
        self.is_1d = is_1d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        Conv = nn.Conv1d if is_1d else nn.Conv2d
        BN = nn.BatchNorm1d if is_1d else nn.BatchNorm2d
        kernel_size = 3
        padding = 1

        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=False)
        self.bn1 = BN(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.sa = SelfAttention(out_channels)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size,
                          stride=1, padding=padding, bias=False)
        self.bn2 = BN(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BN(out_channels)
            )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """ Forward pass through the Pyramidal Residual Block. """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.sa(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.shortcut(identity)
        out += identity
        out = self.relu2(out)
        return out
