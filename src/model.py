# src/model.py
# Defines the main DSSFN model architecture, supporting different fusion mechanisms.

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
        INTERMEDIATE_ATTENTION_STAGES = [] # Default: No intermediate attention for simplicity if config fails
        FUSION_MECHANISM = 'AdaptiveWeight' # Default to test this
        PATCH_SIZE = 15
        SWGMF_TARGET_BANDS = 30 # Example, not directly used by model.py unless passed
        BAND_SELECTION_METHOD = 'SWGMF' # Example
    cfg = MockConfig()
    logging.warning("Could not import config in model.py, using mock defaults for INTERMEDIATE_ATTENTION_STAGES and FUSION_MECHANISM.")


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
                 fusion_mechanism='AdaptiveWeight', 
                 cross_attention_heads=8,
                 cross_attention_dropout=0.1):
        """
        Initializes the DSSFN model.
        Args:
            input_bands (int): Number of input spectral bands (after any band selection).
            num_classes (int): Number of output classes.
            patch_size (int): Spatial size of the input patch (e.g., 15 for 15x15).
            spec_channels (list): List of channel counts for the three stages of the spectral stream.
            spatial_channels (list): List of channel counts for the three stages of the spatial stream.
            fusion_mechanism (str): Type of fusion for final outputs ('AdaptiveWeight' or 'CrossAttention').
            cross_attention_heads (int): Number of heads for MultiHeadCrossAttention modules.
            cross_attention_dropout (float): Dropout rate for MultiHeadCrossAttention modules.
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
        # Use INTERMEDIATE_ATTENTION_STAGES from the imported cfg module
        self.intermediate_stages = cfg.INTERMEDIATE_ATTENTION_STAGES if hasattr(cfg, 'INTERMEDIATE_ATTENTION_STAGES') else []


        # --- Check Channel Compatibility ---
        # Check for intermediate attention stages
        if 1 in self.intermediate_stages and spec_channels[0] != spatial_channels[0]:
            raise ValueError("Intermediate attention after Stage 1 requires spec_channels[0] == spatial_channels[0].")
        if 2 in self.intermediate_stages and spec_channels[1] != spatial_channels[1]:
            raise ValueError("Intermediate attention after Stage 2 requires spec_channels[1] == spatial_channels[1].")
        # Final fusion dimension must match if not using cross-attention fusion that handles different dims before concat
        if spec_channels[2] != spatial_channels[2]:
             raise ValueError(f"Final stage channel dimensions must match: Spec={spec_channels[2]}, Spat={spatial_channels[2]}")
        self.final_fusion_dim = spatial_channels[2] # Used by both fusion mechanisms

        # --- Define Convolutional Layers (needed for size calculations and forward pass) ---
        # Spectral Stream Initial Convolution
        self.spec_conv_in = nn.Conv1d(1, spec_channels[0], kernel_size=3, padding=1, bias=False)
        # Spatial Stream Initial Convolution
        self.spatial_conv_in = nn.Conv2d(input_bands, spatial_channels[0], kernel_size=3, padding=1, bias=False)
        
        # Convolutions between stages (for downsampling/channel increase)
        self.spec_conv1 = nn.Conv1d(spec_channels[0], spec_channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.spec_conv2 = nn.Conv1d(spec_channels[1], spec_channels[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.spatial_conv1 = nn.Conv2d(spatial_channels[0], spatial_channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.spatial_conv2 = nn.Conv2d(spatial_channels[1], spatial_channels[2], kernel_size=3, stride=2, padding=1, bias=False)

        # --- Calculate Sequence Lengths After Each Stage (for positional embeddings) ---
        # Stage 1 output sizes (after initial conv, before PyramidalResidualBlock which maintains size)
        self.spec_len_s1 = _calculate_conv_output_size(self.input_bands, kernel_size=3, stride=1, padding=1) # After spec_conv_in
        self.spat_h_s1 = _calculate_conv_output_size(self.patch_size, kernel_size=3, stride=1, padding=1)    # After spatial_conv_in
        self.spat_w_s1 = self.spat_h_s1
        self.spat_seq_len_s1 = self.spat_h_s1 * self.spat_w_s1

        # Stage 2 output sizes (after stage 1 block and inter-stage conv1)
        self.spec_len_s2 = _calculate_conv_output_size(self.spec_len_s1, kernel_size=3, stride=2, padding=1) # After spec_conv1
        self.spat_h_s2 = _calculate_conv_output_size(self.spat_h_s1, kernel_size=3, stride=2, padding=1)    # After spatial_conv1
        self.spat_w_s2 = self.spat_h_s2
        self.spat_seq_len_s2 = self.spat_h_s2 * self.spat_w_s2
        
        # Stage 3 output sizes (after stage 2 block and inter-stage conv2)
        self.spec_len_s3 = _calculate_conv_output_size(self.spec_len_s2, kernel_size=3, stride=2, padding=1) # After spec_conv2
        self.spat_h_s3 = _calculate_conv_output_size(self.spat_h_s2, kernel_size=3, stride=2, padding=1)    # After spatial_conv2
        self.spat_w_s3 = self.spat_h_s3
        self.spat_seq_len_s3 = self.spat_h_s3 * self.spat_w_s3


        # --- Define Remaining Layers (BN, ReLU, PyramidalResidualBlocks) ---
        # Spectral Stream
        self.spec_bn_in = nn.BatchNorm1d(spec_channels[0])
        self.spec_relu_in = nn.ReLU(inplace=True)
        self.spec_stage1 = PyramidalResidualBlock(spec_channels[0], spec_channels[0], is_1d=True)
        
        self.spec_bn1 = nn.BatchNorm1d(spec_channels[1]) # BN after inter-stage conv1
        # self.spec_relu1 = nn.ReLU(inplace=True) # ReLU is typically after BN, before next block
        self.spec_stage2 = PyramidalResidualBlock(spec_channels[1], spec_channels[1], is_1d=True)
        
        self.spec_bn2 = nn.BatchNorm1d(spec_channels[2]) # BN after inter-stage conv2
        # self.spec_relu2 = nn.ReLU(inplace=True)
        self.spec_stage3 = PyramidalResidualBlock(spec_channels[2], spec_channels[2], is_1d=True)

        # Spatial Stream
        self.spatial_bn_in = nn.BatchNorm2d(spatial_channels[0])
        self.spatial_relu_in = nn.ReLU(inplace=True)
        self.spatial_stage1 = PyramidalResidualBlock(spatial_channels[0], spatial_channels[0], is_1d=False)

        self.spatial_bn1 = nn.BatchNorm2d(spatial_channels[1]) # BN after inter-stage conv1
        # self.spatial_relu1 = nn.ReLU(inplace=True)
        self.spatial_stage2 = PyramidalResidualBlock(spatial_channels[1], spatial_channels[1], is_1d=False)

        self.spatial_bn2 = nn.BatchNorm2d(spatial_channels[2]) # BN after inter-stage conv2
        # self.spatial_relu2 = nn.ReLU(inplace=True)
        self.spatial_stage3 = PyramidalResidualBlock(spatial_channels[2], spatial_channels[2], is_1d=False)


        # --- Intermediate Cross-Attention Modules & Positional Embeddings ---
        self.intermediate_spec_enhancer_s1, self.intermediate_spat_enhancer_s1 = None, None
        self.spec_pos_embedding_s1, self.spat_pos_embedding_s1 = None, None
        self.intermediate_spec_enhancer_s2, self.intermediate_spat_enhancer_s2 = None, None
        self.spec_pos_embedding_s2, self.spat_pos_embedding_s2 = None, None

        if 1 in self.intermediate_stages:
            dim1 = spatial_channels[0] # Channel dimension for stage 1
            self.intermediate_spec_enhancer_s1 = MultiHeadCrossAttention(dim1, cross_attention_heads, cross_attention_dropout)
            self.intermediate_spat_enhancer_s1 = MultiHeadCrossAttention(dim1, cross_attention_heads, cross_attention_dropout)
            # Positional embeddings for features *before* PyramidalResidualBlock of Stage 1
            # The length should match the output of the initial conv layers.
            self.spec_pos_embedding_s1 = nn.Parameter(torch.randn(1, self.spec_len_s1, dim1) * 0.02)
            self.spat_pos_embedding_s1 = nn.Parameter(torch.randn(1, self.spat_seq_len_s1, dim1) * 0.02)
            logging.info(f"DSSFN Intermediate Attention ACTIVE after Stage 1 (Heads: {cross_attention_heads}, Dim: {dim1}).")
            logging.info(f"  Spec Pos Emb S1: (1, {self.spec_len_s1}, {dim1}), Spat Pos Emb S1: (1, {self.spat_seq_len_s1}, {dim1})")

        if 2 in self.intermediate_stages:
            dim2 = spatial_channels[1] # Channel dimension for stage 2
            self.intermediate_spec_enhancer_s2 = MultiHeadCrossAttention(dim2, cross_attention_heads, cross_attention_dropout)
            self.intermediate_spat_enhancer_s2 = MultiHeadCrossAttention(dim2, cross_attention_heads, cross_attention_dropout)
            # Positional embeddings for features *before* PyramidalResidualBlock of Stage 2
            # The length should match the output of the inter-stage conv1 layers.
            self.spec_pos_embedding_s2 = nn.Parameter(torch.randn(1, self.spec_len_s2, dim2) * 0.02)
            self.spat_pos_embedding_s2 = nn.Parameter(torch.randn(1, self.spat_seq_len_s2, dim2) * 0.02)
            logging.info(f"DSSFN Intermediate Attention ACTIVE after Stage 2 (Heads: {cross_attention_heads}, Dim: {dim2}).")
            logging.info(f"  Spec Pos Emb S2: (1, {self.spec_len_s2}, {dim2}), Spat Pos Emb S2: (1, {self.spat_seq_len_s2}, {dim2})")
            
        if not self.intermediate_stages:
             logging.info("DSSFN Intermediate Attention DISABLED.")

        # --- Final Fusion and Classification Layers ---
        if self.fusion_mechanism == 'AdaptiveWeight':
            # Each stream will have its own FC layer leading to logits.
            # The adaptive weighting of these logits will be handled in the training engine.
            self.spec_global_pool = nn.AdaptiveAvgPool1d(1) # Pool to (B, C, 1)
            self.spec_fc = nn.Linear(self.final_fusion_dim, num_classes)
            
            self.spatial_global_pool = nn.AdaptiveAvgPool2d((1, 1)) # Pool to (B, C, 1, 1)
            self.spatial_fc = nn.Linear(self.final_fusion_dim, num_classes)
            logging.info("DSSFN using Adaptive Weight Fusion for FINAL fusion (weights determined by engine based on stream losses).")

        elif self.fusion_mechanism == 'CrossAttention':
            # Final cross-attention fusion before a single classifier
            self.final_spat_enhancer = MultiHeadCrossAttention(self.final_fusion_dim, cross_attention_heads, cross_attention_dropout)
            self.final_spec_enhancer = MultiHeadCrossAttention(self.final_fusion_dim, cross_attention_heads, cross_attention_dropout)
            
            # Positional embeddings for features *before* final cross-attention
            # The length should match the output of the Stage 3 PyramidalResidualBlocks.
            self.spec_pos_embedding_s3 = nn.Parameter(torch.randn(1, self.spec_len_s3, self.final_fusion_dim) * 0.02)
            self.spat_pos_embedding_s3 = nn.Parameter(torch.randn(1, self.spat_seq_len_s3, self.final_fusion_dim) * 0.02)
            
            self.fusion_global_pool = nn.AdaptiveAvgPool1d(1) # Applied after cross-attention and concatenation
            # Input to fusion_fc will be concatenated features from both enhanced streams
            self.fusion_fc = nn.Linear(self.final_fusion_dim * 2, num_classes) 
            logging.info(f"DSSFN using Bidirectional Cross-Attention Fusion ({cross_attention_heads} heads, Dim: {self.final_fusion_dim}) for FINAL fusion.")
            logging.info(f"  Spec Pos Emb S3: (1, {self.spec_len_s3}, {self.final_fusion_dim}), Spat Pos Emb S3: (1, {self.spat_seq_len_s3}, {self.final_fusion_dim})")
        else:
            raise ValueError(f"Unsupported final fusion_mechanism: {self.fusion_mechanism}")

    def _apply_intermediate_attention(self, spc_in, spt_in, stage_num):
        """
        Applies bidirectional cross-attention between spectral and spatial streams.
        Args:
            spc_in (torch.Tensor): Spectral features (B, C, L).
            spt_in (torch.Tensor): Spatial features (B, C, H, W).
            stage_num (int): The stage number (1 or 2) after which attention is applied.
        Returns:
            tuple: (spc_enhanced, spt_enhanced)
        """
        if stage_num == 1:
            spec_enhancer, spat_enhancer = self.intermediate_spec_enhancer_s1, self.intermediate_spat_enhancer_s1
            spec_pos_emb, spat_pos_emb = self.spec_pos_embedding_s1, self.spat_pos_embedding_s1
        elif stage_num == 2:
            spec_enhancer, spat_enhancer = self.intermediate_spec_enhancer_s2, self.intermediate_spat_enhancer_s2
            spec_pos_emb, spat_pos_emb = self.spec_pos_embedding_s2, self.spat_pos_embedding_s2
        else:
            raise ValueError(f"Invalid stage_num for intermediate attention: {stage_num}")

        # Reshape spatial features for attention: (B, C, H, W) -> (B, N_spt, C) where N_spt = H*W
        B, C, H, W = spt_in.shape
        N_spt = H * W
        spt_reshaped = spt_in.view(B, C, N_spt).permute(0, 2, 1).contiguous() # (B, N_spt, C)
        
        # Reshape spectral features for attention: (B, C, L) -> (B, L, C)
        B_spc, C_spc, L_spc = spc_in.shape
        spc_reshaped = spc_in.permute(0, 2, 1).contiguous() # (B, L_spc, C)

        # Add Positional Embeddings, slicing if necessary to match actual sequence lengths
        # This ensures positional embeddings match the sequence length of spc_reshaped/spt_reshaped
        L_slice = min(L_spc, spec_pos_emb.shape[1])
        spc_with_pos = spc_reshaped[:, :L_slice, :] + spec_pos_emb[:, :L_slice, :]
        
        N_slice = min(N_spt, spat_pos_emb.shape[1])
        spt_with_pos = spt_reshaped[:, :N_slice, :] + spat_pos_emb[:, :N_slice, :]

        # Spectral stream enhanced by spatial context
        # Query: spc_with_pos, Context: spt_with_pos
        spc_enhanced_reshaped = spec_enhancer(spc_with_pos, spt_with_pos) # Output: (B, L_slice, C)
        spc_enhanced = spc_enhanced_reshaped.permute(0, 2, 1) # Back to (B, C, L_slice)
        
        # Pad if L_slice was smaller than original L_spc to maintain tensor shape for subsequent layers
        if L_slice < L_spc:
            padding_needed = L_spc - L_slice
            # Pad the last dimension (sequence length)
            spc_enhanced = F.pad(spc_enhanced, (0, padding_needed))


        # Spatial stream enhanced by spectral context
        # Query: spt_with_pos, Context: spc_with_pos
        spt_enhanced_reshaped = spat_enhancer(spt_with_pos, spc_with_pos) # Output: (B, N_slice, C)
        spt_enhanced_permuted = spt_enhanced_reshaped.permute(0, 2, 1) # Back to (B, C, N_slice)

        # Pad if N_slice was smaller than original N_spt
        if N_slice < N_spt:
            padding_needed = N_spt - N_slice
            spt_enhanced_permuted = F.pad(spt_enhanced_permuted, (0, padding_needed))
            
        spt_enhanced = spt_enhanced_permuted.view(B, C, H, W) # Reshape back to (B, C, H, W)

        return spc_enhanced, spt_enhanced

    def forward(self, x_spatial):
        """
        Forward pass of the DSSFN model.
        Args:
            x_spatial (torch.Tensor): Input spatial patch (B, NumInputBands, PatchH, PatchW).
                                      NumInputBands is the number of bands after initial band selection.
        Returns:
            If fusion_mechanism is 'AdaptiveWeight':
                tuple: (spec_logits, spatial_logits) - Logits from each stream.
            If fusion_mechanism is 'CrossAttention':
                torch.Tensor: fused_logits - Logits from the combined stream.
        """
        # --- Initial Feature Extraction ---
        # Spatial Stream
        spt = self.spatial_relu_in(self.spatial_bn_in(self.spatial_conv_in(x_spatial)))
        
        # Spectral Stream: Extract center pixel's spectrum from the spatial patch
        # x_spatial is (B, input_bands, patch_size, patch_size)
        # x_spectral should be (B, 1, input_bands) for Conv1d
        center_pixel_r, center_pixel_c = self.patch_size // 2, self.patch_size // 2
        x_spectral = x_spatial[:, :, center_pixel_r, center_pixel_c].unsqueeze(1) # (B, NumInputBands) -> (B, 1, NumInputBands)
        spc = self.spec_relu_in(self.spec_bn_in(self.spec_conv_in(x_spectral)))

        # --- Stage 1 ---
        # Apply PyramidalResidualBlock first, then intermediate attention if configured
        spt_s1_block_out = self.spatial_stage1(spt)
        spc_s1_block_out = self.spec_stage1(spc)

        if 1 in self.intermediate_stages:
            # Apply attention to the output of the initial conv + BN + ReLU, *before* the first PyramidalResidualBlock
            # Or, apply to the output of the PyramidalResidualBlock. The paper figure (Fig 5) shows
            # attention block within the "Pyramidal ResNet Attention Block".
            # Current modules.py has SA inside PyramidalResidualBlock.
            # The self.intermediate_spec_enhancer is a *cross*-attention, intended *between* streams.
            # Let's assume intermediate cross-attention is applied *after* the Stage 1 blocks.
            spc_s1_out, spt_s1_out = self._apply_intermediate_attention(spc_s1_block_out, spt_s1_block_out, 1)
        else:
            spc_s1_out, spt_s1_out = spc_s1_block_out, spt_s1_block_out
        
        # --- Stage 2 ---
        # Inter-stage convolutions, BN, ReLU
        spt_s2_in = self.spatial_relu_in(self.spatial_bn1(self.spatial_conv1(spt_s1_out)))
        spc_s2_in = self.spec_relu_in(self.spec_bn1(self.spec_conv1(spc_s1_out)))
        
        spt_s2_block_out = self.spatial_stage2(spt_s2_in)
        spc_s2_block_out = self.spec_stage2(spc_s2_in)

        if 2 in self.intermediate_stages:
             spc_s2_out, spt_s2_out = self._apply_intermediate_attention(spc_s2_block_out, spt_s2_block_out, 2)
        else:
             spc_s2_out, spt_s2_out = spc_s2_block_out, spt_s2_block_out

        # --- Stage 3 ---
        # Inter-stage convolutions, BN, ReLU
        spt_s3_in = self.spatial_relu_in(self.spatial_bn2(self.spatial_conv2(spt_s2_out)))
        spc_s3_in = self.spec_relu_in(self.spec_bn2(self.spec_conv2(spc_s2_out)))

        spt_features = self.spatial_stage3(spt_s3_in) # Final spatial features (B, C3, H3, W3)
        spc_features = self.spec_stage3(spc_s3_in)   # Final spectral features (B, C3, L3)


        # --- Apply FINAL Fusion Mechanism ---
        if self.fusion_mechanism == 'AdaptiveWeight':
            # Global pooling and FC layer for spatial stream
            spt_pooled = self.spatial_global_pool(spt_features).flatten(start_dim=1) # Flatten from C onwards
            spatial_logits = self.spatial_fc(spt_pooled)
            
            # Global pooling and FC layer for spectral stream
            spc_pooled = self.spec_global_pool(spc_features).flatten(start_dim=1) # Flatten from C onwards
            spec_logits = self.spec_fc(spc_pooled)
            
            # Return individual logits; actual combination and weighting handled by the engine
            return spec_logits, spatial_logits

        elif self.fusion_mechanism == 'CrossAttention':
            # Reshape features for cross-attention: (B, SeqLen, Dim)
            B, C3_spt, H3, W3 = spt_features.shape
            N3_spt = H3 * W3 # Spatial sequence length
            spt_final_reshaped = spt_features.view(B, C3_spt, N3_spt).permute(0, 2, 1).contiguous() # (B, N3_spt, C3_spt)
            
            B_spc, C3_spc, L3_spc = spc_features.shape # Spectral sequence length is L3_spc
            spc_final_reshaped = spc_features.permute(0, 2, 1).contiguous() # (B, L3_spc, C3_spc)

            # Add Positional Embeddings for final cross-attention
            L3_slice = min(L3_spc, self.spec_pos_embedding_s3.shape[1])
            spc_final_pos = spc_final_reshaped[:, :L3_slice, :] + self.spec_pos_embedding_s3[:, :L3_slice, :]
            
            N3_slice = min(N3_spt, self.spat_pos_embedding_s3.shape[1])
            spt_final_pos = spt_final_reshaped[:, :N3_slice, :] + self.spat_pos_embedding_s3[:, :N3_slice, :]
            
            # Pad if sliced, to ensure original sequence length for enhancers if they expect it
            if L3_slice < spc_final_reshaped.shape[1]:
                 spc_final_pos = F.pad(spc_final_pos, (0,0, 0, spc_final_reshaped.shape[1]-L3_slice )) # Pad seq_len dim
            if N3_slice < spt_final_reshaped.shape[1]:
                 spt_final_pos = F.pad(spt_final_pos, (0,0, 0, spt_final_reshaped.shape[1]-N3_slice )) # Pad seq_len dim


            # Apply bidirectional cross-attention
            # Spectral stream enhanced by spatial context
            fused_spec_q = self.final_spec_enhancer(spc_final_pos, spt_final_pos) # Query: spec, Context: spat
            # Spatial stream enhanced by spectral context
            fused_spat_q = self.final_spat_enhancer(spt_final_pos, spc_final_pos) # Query: spat, Context: spec

            # Global average pooling on the sequence dimension
            # Permute to (B, C, SeqLen) before pooling
            pooled_spec_q = self.fusion_global_pool(fused_spec_q.permute(0, 2, 1)).flatten(start_dim=1)
            pooled_spat_q = self.fusion_global_pool(fused_spat_q.permute(0, 2, 1)).flatten(start_dim=1)

            # Concatenate pooled features and pass through final FC layer
            final_fused_pooled = torch.cat((pooled_spec_q, pooled_spat_q), dim=1) # (B, C3*2)
            fused_logits = self.fusion_fc(final_fused_pooled)
            
            # When not AdaptiveWeight, engine.py expects only one output (logits)
            return fused_logits
        else:
             raise ValueError(f"Unsupported final fusion_mechanism: {self.fusion_mechanism}")

