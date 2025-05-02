# src/model.py
# Defines the main DSSFN model architecture, supporting different fusion mechanisms.
# <<< MODIFICATION: Changed imports for modules and config to relative imports >>>

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

# Import the building blocks from modules.py using relative import
from .modules import PyramidalResidualBlock, MultiHeadCrossAttention
# Import config to read parameters using relative import
try:
    from . import config as cfg
except ImportError:
    # Provide default if run standalone or config fails
    class MockConfig:
        INTERMEDIATE_ATTENTION_STAGES = [1] # Default: Stage 1 only
        FUSION_MECHANISM = 'CrossAttention' # Default to test this
        PATCH_SIZE = 15
        SWGMF_TARGET_BANDS = 30
        BAND_SELECTION_METHOD = 'SWGMF'
    cfg = MockConfig()
    logging.warning("Could not import config in model.py, using defaults.")


def _calculate_conv_output_size(input_size, kernel_size, stride, padding):
    """ Calculates the output size of a Conv1D or Conv2D dimension. """
    # Ensure input_size is an integer
    input_size = int(input_size)
    return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1

class DSSFN(nn.Module):
    """
    Dual-Stream Self-Attention Fusion Network (DSSFN).
    Supports Adaptive Weighted Fusion or a new Bidirectional Cross-Attention Fusion for final output.
    Includes optional, configurable BIDIRECTIONAL intermediate cross-attention.
    """
    def __init__(self, input_bands, num_classes, patch_size,
                 spec_channels=[64, 128, 256], spatial_channels=[64, 128, 256],
                 fusion_mechanism='CrossAttention', # Defaulting to CrossAttention to test new fusion
                 cross_attention_heads=8,
                 cross_attention_dropout=0.1):
        """
        Initializes the DSSFN model.
        """
        super(DSSFN, self).__init__()

        # --- Basic Setup & Checks ---
        if len(spec_channels) != 3 or len(spatial_channels) != 3:
            raise ValueError("Channel lists for spectral and spatial streams must have length 3.")

        self.input_bands = input_bands # AFTER band selection
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.spec_channels = spec_channels
        self.spatial_channels = spatial_channels
        self.fusion_mechanism = fusion_mechanism
        self.intermediate_stages = cfg.INTERMEDIATE_ATTENTION_STAGES

        # --- Check Channel Compatibility ---
        if 1 in self.intermediate_stages and spec_channels[0] != spatial_channels[0]:
            raise ValueError("Intermediate attention after Stage 1 requires spec_channels[0] == spatial_channels[0].")
        if 2 in self.intermediate_stages and spec_channels[1] != spatial_channels[1]:
            raise ValueError("Intermediate attention after Stage 2 requires spec_channels[1] == spatial_channels[1].")
        # Final fusion check (both AdaptiveWeight and CrossAttention need this now)
        if spec_channels[2] != spatial_channels[2]:
             raise ValueError(f"Final stage channel dimensions must match: Spec={spec_channels[2]}, Spat={spatial_channels[2]}")
        self.final_fusion_dim = spatial_channels[2] # Dimension for final fusion

        # --- Define Convolutional Layers (needed for size calculations) ---
        # Spectral Stream Conv Layers
        self.spec_conv_in = nn.Conv1d(1, spec_channels[0], kernel_size=3, padding=1, bias=False) # k=3, s=1, p=1
        self.spec_conv1 = nn.Conv1d(spec_channels[0], spec_channels[1], kernel_size=3, stride=2, padding=1, bias=False) # k=3, s=2, p=1
        self.spec_conv2 = nn.Conv1d(spec_channels[1], spec_channels[2], kernel_size=3, stride=2, padding=1, bias=False) # k=3, s=2, p=1
        # Spatial Stream Conv Layers
        self.spatial_conv_in = nn.Conv2d(input_bands, spatial_channels[0], kernel_size=3, padding=1, bias=False) # k=3, s=1, p=1
        self.spatial_conv1 = nn.Conv2d(spatial_channels[0], spatial_channels[1], kernel_size=3, stride=2, padding=1, bias=False) # k=3, s=2, p=1
        self.spatial_conv2 = nn.Conv2d(spatial_channels[1], spatial_channels[2], kernel_size=3, stride=2, padding=1, bias=False) # k=3, s=2, p=1

        # --- Calculate Sequence Lengths After Each Stage ---
        # Stage 1
        self.spec_len_s1 = _calculate_conv_output_size(self.input_bands, kernel_size=3, stride=1, padding=1)
        self.spat_h_s1 = _calculate_conv_output_size(self.patch_size, kernel_size=3, stride=1, padding=1)
        self.spat_w_s1 = self.spat_h_s1
        self.spat_seq_len_s1 = self.spat_h_s1 * self.spat_w_s1
        # Stage 2
        self.spec_len_s2 = _calculate_conv_output_size(self.spec_len_s1, kernel_size=3, stride=2, padding=1)
        self.spat_h_s2 = _calculate_conv_output_size(self.spat_h_s1, kernel_size=3, stride=2, padding=1)
        self.spat_w_s2 = self.spat_h_s2
        self.spat_seq_len_s2 = self.spat_h_s2 * self.spat_w_s2
        # Stage 3
        self.spec_len_s3 = _calculate_conv_output_size(self.spec_len_s2, kernel_size=3, stride=2, padding=1)
        self.spat_h_s3 = _calculate_conv_output_size(self.spat_h_s2, kernel_size=3, stride=2, padding=1)
        self.spat_w_s3 = self.spat_h_s3
        self.spat_seq_len_s3 = self.spat_h_s3 * self.spat_w_s3

        # --- Define Remaining Layers ---
        # Spectral Stream
        self.spec_bn_in = nn.BatchNorm1d(spec_channels[0])
        self.spec_relu_in = nn.ReLU(inplace=True)
        self.spec_stage1 = PyramidalResidualBlock(spec_channels[0], spec_channels[0], is_1d=True)
        self.spec_bn1 = nn.BatchNorm1d(spec_channels[1])
        self.spec_stage2 = PyramidalResidualBlock(spec_channels[1], spec_channels[1], is_1d=True)
        self.spec_bn2 = nn.BatchNorm1d(spec_channels[2])
        self.spec_stage3 = PyramidalResidualBlock(spec_channels[2], spec_channels[2], is_1d=True)
        # Spatial Stream
        self.spatial_bn_in = nn.BatchNorm2d(spatial_channels[0])
        self.spatial_relu_in = nn.ReLU(inplace=True)
        self.spatial_stage1 = PyramidalResidualBlock(spatial_channels[0], spatial_channels[0], is_1d=False)
        self.spatial_bn1 = nn.BatchNorm2d(spatial_channels[1])
        self.spatial_stage2 = PyramidalResidualBlock(spatial_channels[1], spatial_channels[1], is_1d=False)
        self.spatial_bn2 = nn.BatchNorm2d(spatial_channels[2])
        self.spatial_stage3 = PyramidalResidualBlock(spatial_channels[2], spatial_channels[2], is_1d=False)

        # --- Intermediate Cross-Attention Modules & Positional Embeddings (Conditional) ---
        self.intermediate_spec_enhancer_s1 = None
        self.intermediate_spat_enhancer_s1 = None
        self.spec_pos_embedding_s1 = None
        self.spat_pos_embedding_s1 = None
        self.intermediate_spec_enhancer_s2 = None
        self.intermediate_spat_enhancer_s2 = None
        self.spec_pos_embedding_s2 = None
        self.spat_pos_embedding_s2 = None

        if 1 in self.intermediate_stages:
            dim1 = spatial_channels[0]
            if dim1 % cross_attention_heads != 0: logging.warning(f"Stage 1 intermediate fusion dim ({dim1}) not divisible by heads ({cross_attention_heads}).")
            self.intermediate_spec_enhancer_s1 = MultiHeadCrossAttention(dim1, cross_attention_heads, cross_attention_dropout)
            self.intermediate_spat_enhancer_s1 = MultiHeadCrossAttention(dim1, cross_attention_heads, cross_attention_dropout)
            self.spec_pos_embedding_s1 = nn.Parameter(torch.randn(1, self.spec_len_s1, dim1) * 0.02)
            self.spat_pos_embedding_s1 = nn.Parameter(torch.randn(1, self.spat_seq_len_s1, dim1) * 0.02)
            logging.info(f"DSSFN Intermediate Attention ACTIVE after Stage 1 (Heads: {cross_attention_heads}). PosEmb sizes: Spec={self.spec_len_s1}, Spat={self.spat_seq_len_s1}")

        if 2 in self.intermediate_stages:
            dim2 = spatial_channels[1]
            if dim2 % cross_attention_heads != 0: logging.warning(f"Stage 2 intermediate fusion dim ({dim2}) not divisible by heads ({cross_attention_heads}).")
            self.intermediate_spec_enhancer_s2 = MultiHeadCrossAttention(dim2, cross_attention_heads, cross_attention_dropout)
            self.intermediate_spat_enhancer_s2 = MultiHeadCrossAttention(dim2, cross_attention_heads, cross_attention_dropout)
            self.spec_pos_embedding_s2 = nn.Parameter(torch.randn(1, self.spec_len_s2, dim2) * 0.02)
            self.spat_pos_embedding_s2 = nn.Parameter(torch.randn(1, self.spat_seq_len_s2, dim2) * 0.02)
            logging.info(f"DSSFN Intermediate Attention ACTIVE after Stage 2 (Heads: {cross_attention_heads}). PosEmb sizes: Spec={self.spec_len_s2}, Spat={self.spat_seq_len_s2}")

        if not self.intermediate_stages:
             logging.info("DSSFN Intermediate Attention DISABLED.")

        # --- Final Fusion and Classification Layers ---
        self.final_spat_enhancer = None
        self.final_spec_enhancer = None
        self.spec_pos_embedding_s3 = None
        self.spat_pos_embedding_s3 = None
        self.fusion_global_pool = None
        self.fusion_fc = None
        self.spec_global_pool = None
        self.spec_fc = None
        self.spatial_global_pool = None
        self.spatial_fc = None

        if self.fusion_mechanism == 'AdaptiveWeight':
            self.spec_global_pool = nn.AdaptiveAvgPool1d(1)
            self.spec_fc = nn.Linear(self.final_fusion_dim, num_classes)
            self.spatial_global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.spatial_fc = nn.Linear(self.final_fusion_dim, num_classes)
            logging.info("DSSFN using Adaptive Weight Fusion (External) for FINAL fusion.")

        elif self.fusion_mechanism == 'CrossAttention':
            if self.final_fusion_dim % cross_attention_heads != 0:
                 logging.warning(f"Final fusion dim ({self.final_fusion_dim}) not divisible by heads ({cross_attention_heads}).")
            self.final_spat_enhancer = MultiHeadCrossAttention(
                in_dim=self.final_fusion_dim, num_heads=cross_attention_heads, dropout=cross_attention_dropout
            )
            self.final_spec_enhancer = MultiHeadCrossAttention(
                in_dim=self.final_fusion_dim, num_heads=cross_attention_heads, dropout=cross_attention_dropout
            )
            self.spec_pos_embedding_s3 = nn.Parameter(torch.randn(1, self.spec_len_s3, self.final_fusion_dim) * 0.02)
            self.spat_pos_embedding_s3 = nn.Parameter(torch.randn(1, self.spat_seq_len_s3, self.final_fusion_dim) * 0.02)
            self.fusion_global_pool = nn.AdaptiveAvgPool1d(1) # Pool over sequence length
            self.fusion_fc = nn.Linear(self.final_fusion_dim * 2, num_classes) # Input size is doubled
            logging.info(f"DSSFN using Bidirectional Cross-Attention Fusion ({cross_attention_heads} heads) for FINAL fusion.")
        else:
            raise ValueError(f"Unsupported final fusion_mechanism: {self.fusion_mechanism}")


    def _apply_intermediate_attention(self, spc_in, spt_in, stage_num):
        """ Helper function to apply bidirectional attention for a given intermediate stage """
        if stage_num == 1:
            spec_enhancer, spat_enhancer = self.intermediate_spec_enhancer_s1, self.intermediate_spat_enhancer_s1
            spec_pos_emb, spat_pos_emb = self.spec_pos_embedding_s1, self.spat_pos_embedding_s1
        elif stage_num == 2:
            spec_enhancer, spat_enhancer = self.intermediate_spec_enhancer_s2, self.intermediate_spat_enhancer_s2
            spec_pos_emb, spat_pos_emb = self.spec_pos_embedding_s2, self.spat_pos_embedding_s2
        else:
            raise ValueError(f"Invalid stage_num for intermediate attention: {stage_num}")

        # Reshape
        B, C, H, W = spt_in.shape
        N = H * W
        spt_reshaped = spt_in.view(B, C, N).permute(0, 2, 1) # (B, N, C)
        _, _, L = spc_in.shape
        spc_reshaped = spc_in.permute(0, 2, 1) # (B, L, C)

        # Add Positional Embeddings
        L_slice = min(L, spec_pos_emb.shape[1])
        if L > spec_pos_emb.shape[1]: logging.warning(f"Stage {stage_num}: Actual spectral length L ({L}) > embedding size ({spec_pos_emb.shape[1]}). Truncating.")
        spc_reshaped = spc_reshaped[:, :L_slice, :] + spec_pos_emb[:, :L_slice, :] # Slice input if needed

        N_slice = min(N, spat_pos_emb.shape[1])
        if N > spat_pos_emb.shape[1]: logging.warning(f"Stage {stage_num}: Actual spatial length N ({N}) > embedding size ({spat_pos_emb.shape[1]}). Truncating.")
        spt_reshaped = spt_reshaped[:, :N_slice, :] + spat_pos_emb[:, :N_slice, :] # Slice input if needed

        # Apply Attention
        spc_enhanced_reshaped = spec_enhancer(spc_reshaped, spt_reshaped)
        spc_enhanced = spc_enhanced_reshaped.permute(0, 2, 1) # (B, C, L_slice)

        spt_enhanced_reshaped = spat_enhancer(spt_reshaped, spc_reshaped)
        # Ensure enhanced spatial features are reshaped correctly even if N was sliced
        # We need to reconstruct H, W based on N_slice if possible, or handle potential mismatch.
        # For simplicity, assume H, W correspond to the *original* N before potential slicing for now.
        # A more robust solution might involve padding or ensuring embeddings match max possible N.
        spt_enhanced = spt_enhanced_reshaped.permute(0, 2, 1).view(B, C, H, W) # (B, C, H, W)

        return spc_enhanced, spt_enhanced


    def forward(self, x_spatial):
        """
        Forward pass with configurable intermediate attention stages and final fusion.
        """
        # --- Initial Feature Extraction ---
        spt = self.spatial_relu_in(self.spatial_bn_in(self.spatial_conv_in(x_spatial)))
        center_pixel_r, center_pixel_c = self.patch_size // 2, self.patch_size // 2
        x_spectral = x_spatial[:, :, center_pixel_r, center_pixel_c].unsqueeze(1)
        spc = self.spec_relu_in(self.spec_bn_in(self.spec_conv_in(x_spectral)))

        # --- Stage 1 ---
        spt_s1_out = self.spatial_stage1(spt)
        spc_s1_out = self.spec_stage1(spc)
        if 1 in self.intermediate_stages:
            spc_cur, spt_cur = self._apply_intermediate_attention(spc_s1_out, spt_s1_out, 1)
        else:
            spc_cur, spt_cur = spc_s1_out, spt_s1_out

        # --- Stage 2 ---
        spt = self.spatial_relu_in(self.spatial_bn1(self.spatial_conv1(spt_cur)))
        spt_s2_out = self.spatial_stage2(spt)
        spc = self.spec_relu_in(self.spec_bn1(self.spec_conv1(spc_cur)))
        spc_s2_out = self.spec_stage2(spc)
        if 2 in self.intermediate_stages:
             spc_cur, spt_cur = self._apply_intermediate_attention(spc_s2_out, spt_s2_out, 2)
        else:
             spc_cur, spt_cur = spc_s2_out, spt_s2_out

        # --- Stage 3 ---
        spt = self.spatial_relu_in(self.spatial_bn2(self.spatial_conv2(spt_cur)))
        spt_features = self.spatial_stage3(spt) # Final spatial features (B, C3, H3, W3)
        spc = self.spec_relu_in(self.spec_bn2(self.spec_conv2(spc_cur)))
        spc_features = self.spec_stage3(spc) # Final spectral features (B, C3, L3)

        # --- Apply FINAL Fusion Mechanism ---
        if self.fusion_mechanism == 'AdaptiveWeight':
            spt_pooled = self.spatial_global_pool(spt_features).flatten(1)
            spatial_logits = self.spatial_fc(spt_pooled)
            spc_pooled = self.spec_global_pool(spc_features).flatten(1)
            spec_logits = self.spec_fc(spc_pooled)
            return spec_logits, spatial_logits

        elif self.fusion_mechanism == 'CrossAttention':
            # Reshape final features
            B, C3, H3, W3 = spt_features.shape
            N3 = H3 * W3
            spt_final_reshaped = spt_features.view(B, C3, N3).permute(0, 2, 1) # (B, N3, C3)
            _, _, L3 = spc_features.shape
            spc_final_reshaped = spc_features.permute(0, 2, 1) # (B, L3, C3)

            # Add final positional embeddings
            L3_slice = min(L3, self.spec_pos_embedding_s3.shape[1])
            if L3 > self.spec_pos_embedding_s3.shape[1]: logging.warning(f"Final Stage: Actual spectral length L ({L3}) > embedding size ({self.spec_pos_embedding_s3.shape[1]}). Truncating.")
            spc_final_reshaped = spc_final_reshaped[:, :L3_slice, :] + self.spec_pos_embedding_s3[:, :L3_slice, :]

            N3_slice = min(N3, self.spat_pos_embedding_s3.shape[1])
            if N3 > self.spat_pos_embedding_s3.shape[1]: logging.warning(f"Final Stage: Actual spatial length N ({N3}) > embedding size ({self.spat_pos_embedding_s3.shape[1]}). Truncating.")
            spt_final_reshaped = spt_final_reshaped[:, :N3_slice, :] + self.spat_pos_embedding_s3[:, :N3_slice, :]

            # Apply bidirectional final cross-attention
            fused_spec_q = self.final_spec_enhancer(spc_final_reshaped, spt_final_reshaped) # (B, L3_slice, C3)
            fused_spat_q = self.final_spat_enhancer(spt_final_reshaped, spc_final_reshaped) # (B, N3_slice, C3)

            # Pool each attention output separately
            pooled_spec_q = self.fusion_global_pool(fused_spec_q.permute(0, 2, 1)).flatten(1) # (B, C3)
            pooled_spat_q = self.fusion_global_pool(fused_spat_q.permute(0, 2, 1)).flatten(1) # (B, C3)

            # Concatenate pooled features
            final_fused_pooled = torch.cat((pooled_spec_q, pooled_spat_q), dim=1) # (B, 2*C3)

            # Final classification 
            fused_logits = self.fusion_fc(final_fused_pooled)
            return fused_logits
        else:
             raise ValueError(f"Unsupported final fusion_mechanism: {self.fusion_mechanism}")
