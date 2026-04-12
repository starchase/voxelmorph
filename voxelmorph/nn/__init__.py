"""
Torch-based neural network components for VoxelMorph. This subpackage contains the functional
operators, reusable building blocks, model definitions, and loss functions to implement the
VoxelMorph framework in PyTorch.

Modules
-------
functional
    Functions containing the core operations and logic of for image registration written in
    PyTorch.
losses
    Loss functions for image registration.
models
    Core VoxelMorph models for unsupervised and supervised learning.
modules
    Neural network building blocks for VoxelMorph.
"""

from . import functional
from . import losses
from . import models
from . import modules
from . import fpn
from . import adaptive_fda

__all__ = [
    "functional",
    "losses",
    "models",
    "modules",
    "fpn",
    "adaptive_fda",
]
from .fpn import VxmFPN, FPNDecoder, SimpleFeatureExtractor
from .fpn_siamese import SiameseFeatureExtractor
from .siamese_baseline import SiameseUNetBaseline
from .adaptive_fda import AdaptiveFDA3D
