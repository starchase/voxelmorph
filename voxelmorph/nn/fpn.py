import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, ndim):
        super().__init__()
        ConvBlock = getattr(nn, f'Conv{ndim}d')
        self.ndim = ndim
        
        # 1x1 conv to match channels if skip connection exists
        self.skip_conv = ConvBlock(skip_channels, out_channels, kernel_size=1) if skip_channels > 0 else None
        
        # 3x3 conv to smooth the upsampled features
        self.smooth_conv = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 3x3 conv for final output of this block
        self.out_conv = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x, skip=None):
        # Upsample using nearest neighbor or trilinear depending on dimension
        # Actually standard voxelmorph uses UpSample layer, we will just use interpolate
        mode = 'trilinear' if self.ndim == 3 else 'bilinear'
        x = F.interpolate(x, scale_factor=2.0, mode=mode, align_corners=False)
        x = self.act(self.smooth_conv(x))
        if skip is not None and self.skip_conv is not None:
            skip = self.act(self.skip_conv(skip))
            x = x + skip
        x = self.act(self.out_conv(x))
        return x

class SimpleFeatureExtractor(nn.Module):
    """
    A simple UNet-like encoder as the default backbone.
    Takes in concatenated source and target images.
    """
    def __init__(self, in_channels, enc_nf, ndim):
        super().__init__()
        ConvBlock = getattr(nn, f'Conv{ndim}d')
        self.enc_blocks = nn.ModuleList()
        
        prev_channels = in_channels
        for nf in enc_nf:
            self.enc_blocks.append(
                nn.Sequential(
                    ConvBlock(prev_channels, nf, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2)
                )
            )
            prev_channels = nf
            
    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
        return features

class FPNDecoder(nn.Module):
    def __init__(self, feature_dims, dec_nf, ndim):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.flow_convs = nn.ModuleList()
        self.ndim = ndim
        
        from .modules import SpatialTransformer
        self.spatial_transform = SpatialTransformer()
        ConvBlock = getattr(nn, f'Conv{ndim}d')
        
        # feature_dims matches the output channels of the encoder blocks [feat0, feat1, ..., featN]
        # dec_nf is the number of channels for each decoder block [dec0, dec1, ...]
        
        # Start from the finest resolution level (last feature from encoder)
        prev_channels = feature_dims[-1]
        
        for i, nf in enumerate(dec_nf):
            # The skip connection comes from the encoder level matching the current resolution
            # feature_dims is e.g. [16, 32, 32, 32]. 
            # In UNet, we upsample from the last level and concat/add with the previous.
            # So the skip index would be len(feature_dims) - 2 - i
            skip_idx = len(feature_dims) - 2 - i
            skip_channels = feature_dims[skip_idx] if skip_idx >= 0 else 0
            
            self.blocks.append(FPNBlock(prev_channels, skip_channels, nf, ndim))
            
            # Flow field prediction sub-layer
            flow_conv = ConvBlock(nf, ndim, kernel_size=3, padding=1)
            flow_conv.weight.data.normal_(0, 1e-5)
            flow_conv.bias.data.zero_()
            self.flow_convs.append(flow_conv)

            prev_channels = nf
            
    def forward(self, features):
        x = features[-1]
        flow = None
        
        for i, block in enumerate(self.blocks):
            skip_idx = len(features) - 2 - i
            skip = features[skip_idx] if skip_idx >= 0 else None
            
            # 1. Block processing
            x = block(x, skip)
            
            # 2. Preliminary deformation sub-field of this layer
            flow_pre = self.flow_convs[i](x)
            
            # 3. Accumulate with previous layer's flow
            if flow is None:
                flow = flow_pre
            else:
                # Upsample previous flow (scale spatially and magnitude)
                mode = 'trilinear' if self.ndim == 3 else 'bilinear'
                flow_up = F.interpolate(flow, scale_factor=2.0, mode=mode, align_corners=False)
                flow_up = flow_up * 2.0
                
                # Warp the previous upsampled flow using the preliminary flow
                warped_flow_up = self.spatial_transform(flow_up, flow_pre)
                
                # Combine the warped previous flow with the preliminary flow
                flow = warped_flow_up + flow_pre
                
        return x, flow

class VxmFPN(nn.Module):
    """
    Decoupled VoxelMorph model using a feature pyramid network structure.
    This allows easy replacement of the backbone for ablation studies.
    """
    def __init__(self, 
                 inshape, 
                 nb_unet_features=None,
                 backbone=None,
                 backbone_feature_dims=None,
                 int_steps=7, 
                 int_downsize=2, 
                 bidir=False,
                 use_probabilities=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        super().__init__()
        
        self.inshape = inshape
        self.int_steps = int_steps
        self.int_downsize = int_downsize
        self.bidir = bidir
        
        ndim = len(inshape)
        
        # Default features if none provided
        if nb_unet_features is None:
            from ..py.utils import default_unet_features
            nb_unet_features = default_unet_features()
        enc_nf = nb_unet_features[0]
        dec_nf = nb_unet_features[1]
        
        if backbone is None:
            self.backbone = SimpleFeatureExtractor(src_feats + trg_feats, enc_nf, ndim)
            self.feature_dims = enc_nf
        else:
            self.backbone = backbone
            if backbone_feature_dims is None:
                raise ValueError("backbone_feature_dims must be provided if using a custom backbone")
            self.feature_dims = backbone_feature_dims
            
        self.decoder = FPNDecoder(self.feature_dims, dec_nf, ndim)
        
        from .modules import SpatialTransformer, IntegrateVelocityField
        
        self.spatial_transform = SpatialTransformer()
        if self.int_steps > 0:
            self.integrate = IntegrateVelocityField(steps=self.int_steps)
        else:
            self.integrate = None
            

    def forward(self, source, target, return_warped_source=False, return_warped_target=False, return_field_type='displacement'):
        # 1. Feature Extraction (Backbone)
        # Handle Siamese dual-stream backbone that takes separate inputs
        if hasattr(self.backbone, 'source_encoder') or hasattr(self.backbone, 'target_encoder'):
            features = self.backbone(source, target)
        else:
            # Standard VoxelMorph backbone expects concatenated inputs
            x = torch.cat([source, target], dim=1)
            features = self.backbone(x)
        
        # 2. Decoder (FPN) with built-in multiscale flow warping
        x, velocity = self.decoder(features)
        
        # Build outputs like VxmPairwise
        pos_displacement = velocity
        if self.integrate is not None:
            pos_displacement = self.integrate(velocity)
            
        outputs = []
        if return_field_type == 'displacement':
            outputs.append(pos_displacement)
        else:
            outputs.append(velocity)
            
        if return_warped_source:
            y_source = self.spatial_transform(source, pos_displacement)
            outputs.append(y_source)
            
        if return_warped_target:
            if self.integrate is not None:
                neg_displacement = self.integrate(-velocity)
            else:
                neg_displacement = -velocity
            y_target = self.spatial_transform(target, neg_displacement)
            outputs.append(y_target)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]
