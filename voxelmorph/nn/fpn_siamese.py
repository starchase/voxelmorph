import torch
import torch.nn as nn
from .fpn import SimpleFeatureExtractor

class SiameseFeatureExtractor(nn.Module):
    """
    Siamese dual-stream backbone that extracts features from source and target images independently.
    Outputs fused features for the FPN Decoder, preserving skip connections.
    """
    def __init__(self, in_channels, enc_nf, ndim, fusion_method='concat'):
        super().__init__()
        self.fusion_method = fusion_method
        assert fusion_method in ['concat', 'add', 'compress_concat'], "fusion_method must be 'concat', 'add', or 'compress_concat'"
        self.output_feature_dims = []
        
        # We instantiate two separate feature extractors (could also share weights if desired)
        # Here we use independent weights, which is often better for multi-modal (CT/MR).
        # For single-modal like OASIS, weight sharing could also be used (by pointing both to the same module).
        self.source_encoder = SimpleFeatureExtractor(in_channels, enc_nf, ndim)
        self.target_encoder = SimpleFeatureExtractor(in_channels, enc_nf, ndim)
        
        # Setup fusion blocks depending on the method
        self.fusion_blocks = nn.ModuleList()
        ConvBlock = getattr(nn, f'Conv{ndim}d')
        
        if fusion_method == 'compress_concat':
            self.source_compress = nn.ModuleList()
            self.target_compress = nn.ModuleList()
        
        for nf in enc_nf:
            if fusion_method == 'concat':
                # Pure concat: no 1x1 projection, decoder must accept doubled channels
                self.output_feature_dims.append(nf * 2)
            elif fusion_method == 'compress_concat':
                # Compress each feature separately, e.g. 64 -> 32, then concat back to nf
                half_nf = max(nf // 2, 1)
                self.source_compress.append(nn.Sequential(ConvBlock(nf, half_nf, kernel_size=1), nn.LeakyReLU(0.2)))
                self.target_compress.append(nn.Sequential(ConvBlock(nf, nf - half_nf, kernel_size=1), nn.LeakyReLU(0.2)))
                self.output_feature_dims.append(nf)
            else:
                self.output_feature_dims.append(nf)

    def forward(self, source, target):
        # source and target are separated here (not concatenated in dim 1 like standard VoxelMorph)
        
        source_feats = self.source_encoder(source)
        target_feats = self.target_encoder(target)
        
        fused_features = []
        for i, (sf, tf) in enumerate(zip(source_feats, target_feats)):
            if self.fusion_method == 'concat':
                fused = torch.cat([sf, tf], dim=1)
            elif self.fusion_method == 'compress_concat':
                sf_c = self.source_compress[i](sf)
                tf_c = self.target_compress[i](tf)
                fused = torch.cat([sf_c, tf_c], dim=1)
            elif self.fusion_method == 'add':
                fused = sf + tf
            fused_features.append(fused)
            
        return fused_features
