"""
Unit tests for the AdaptiveFDA3D module.
"""

# Standard library imports
import math

# Third-party imports
import pytest
import torch

# Local imports
import voxelmorph as vxm
from voxelmorph.nn.adaptive_fda import AdaptiveFDA3D, _soft_low_freq_mask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vol_pair():
    """Small (B=1, C=1, 8x8x8) source/target pair in [0,1]."""
    torch.manual_seed(0)
    source = torch.rand(1, 1, 8, 8, 8)
    target = torch.rand(1, 1, 8, 8, 8)
    return source, target


# ---------------------------------------------------------------------------
# _soft_low_freq_mask helper
# ---------------------------------------------------------------------------

class TestSoftLowFreqMask:
    def test_shape(self):
        mask = _soft_low_freq_mask(8, 8, 8, torch.tensor(0.1))
        assert mask.shape == (1, 1, 8, 8, 8)

    def test_range(self):
        mask = _soft_low_freq_mask(8, 8, 8, torch.tensor(0.2))
        assert mask.min().item() >= 0.0
        assert mask.max().item() <= 1.0

    def test_center_high_periphery_low(self):
        """DC centre should be close to 1; corners should be close to 0."""
        mask = _soft_low_freq_mask(16, 16, 16, torch.tensor(0.15), temperature=0.005)
        # centre voxel (after fftshift the DC is at index D//2, H//2, W//2)
        center_val = mask[0, 0, 8, 8, 8].item()
        corner_val = mask[0, 0, 0, 0, 0].item()
        assert center_val > 0.9, f"Centre value too low: {center_val}"
        assert corner_val < 0.1, f"Corner value too high: {corner_val}"


# ---------------------------------------------------------------------------
# AdaptiveFDA3D – construction
# ---------------------------------------------------------------------------

class TestAdaptiveFDA3DConstruction:
    @pytest.mark.parametrize("mode", ["fixed", "learnable", "adaptive"])
    def test_instantiation(self, mode):
        fda = AdaptiveFDA3D(beta_init=0.1, mode=mode)
        assert isinstance(fda, torch.nn.Module)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            AdaptiveFDA3D(mode="invalid")

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta_init must be"):
            AdaptiveFDA3D(beta_init=0.6)

    def test_fixed_has_no_trainable_params(self):
        fda = AdaptiveFDA3D(beta_init=0.1, mode="fixed")
        assert sum(p.numel() for p in fda.parameters()) == 0

    def test_learnable_has_one_trainable_param(self):
        fda = AdaptiveFDA3D(beta_init=0.1, mode="learnable")
        params = list(fda.parameters())
        assert len(params) == 1
        assert params[0].shape == ()  # scalar

    def test_adaptive_has_multiple_params(self):
        fda = AdaptiveFDA3D(beta_init=0.1, mode="adaptive")
        assert sum(p.numel() for p in fda.parameters()) > 1


# ---------------------------------------------------------------------------
# AdaptiveFDA3D – get_beta
# ---------------------------------------------------------------------------

class TestGetBeta:
    def test_fixed_returns_init_value(self):
        fda = AdaptiveFDA3D(beta_init=0.2, mode="fixed")
        beta = fda.get_beta()
        assert abs(beta.item() - 0.2) < 1e-5

    def test_learnable_beta_in_range(self):
        fda = AdaptiveFDA3D(beta_init=0.1, mode="learnable")
        beta = fda.get_beta()
        assert 0.0 < beta.item() < 0.5

    def test_learnable_beta_near_init(self):
        """Freshly constructed learnable beta should be close to beta_init."""
        fda = AdaptiveFDA3D(beta_init=0.15, mode="learnable")
        beta = fda.get_beta()
        assert abs(beta.item() - 0.15) < 0.02

    def test_adaptive_beta_in_range(self):
        fda = AdaptiveFDA3D(beta_init=0.1, mode="adaptive")
        torch.manual_seed(42)
        src_stats = torch.rand(2, 1)
        tgt_stats = torch.rand(2, 1)
        beta = fda.get_beta(src_stats, tgt_stats)
        assert beta.shape == (2, 1)
        assert (beta > 0).all() and (beta < 0.5).all()

    def test_adaptive_missing_stats_raises(self):
        fda = AdaptiveFDA3D(mode="adaptive")
        with pytest.raises(ValueError, match="required for adaptive mode"):
            fda.get_beta()


# ---------------------------------------------------------------------------
# AdaptiveFDA3D – forward
# ---------------------------------------------------------------------------

class TestAdaptiveFDA3DForward:
    @pytest.mark.parametrize("mode", ["fixed", "learnable", "adaptive"])
    def test_output_shape(self, vol_pair, mode):
        source, target = vol_pair
        fda = AdaptiveFDA3D(beta_init=0.1, mode=mode)
        out = fda(source, target)
        assert out.shape == source.shape

    @pytest.mark.parametrize("mode", ["fixed", "learnable", "adaptive"])
    def test_output_range(self, vol_pair, mode):
        """Output should stay in [0, 1] since input is in [0, 1]."""
        source, target = vol_pair
        fda = AdaptiveFDA3D(beta_init=0.1, mode=mode)
        out = fda(source, target)
        assert out.min().item() >= 0.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    @pytest.mark.parametrize("mode", ["fixed", "learnable", "adaptive"])
    def test_output_dtype_preserved(self, vol_pair, mode):
        """Output dtype should match source dtype."""
        source, target = vol_pair
        fda = AdaptiveFDA3D(beta_init=0.1, mode=mode)
        out = fda(source, target)
        assert out.dtype == source.dtype

    def test_zero_beta_identity(self):
        """With beta≈0 no frequencies are swapped; adapted ≈ source."""
        fda = AdaptiveFDA3D(beta_init=1e-4, mode="fixed")
        torch.manual_seed(7)
        source = torch.rand(1, 1, 8, 8, 8)
        target = torch.rand(1, 1, 8, 8, 8)
        out = fda(source, target)
        # Very small beta → mask ≈ 0 everywhere → output ≈ source
        assert torch.allclose(out, source, atol=1e-3), (
            f"Expected output close to source for beta≈0, max diff: {(out - source).abs().max():.4f}"
        )

    def test_maximum_beta_matches_target_mean_amplitude(self):
        """With beta≈0.5 most amplitudes are swapped towards the target."""
        fda = AdaptiveFDA3D(beta_init=0.49, mode="fixed")
        torch.manual_seed(0)
        source = torch.rand(1, 1, 8, 8, 8)
        target = torch.rand(1, 1, 8, 8, 8)
        out = fda(source, target)
        # Output should differ from source (amplitudes shifted towards target)
        assert not torch.allclose(out, source, atol=1e-3)

    def test_learnable_gradient_flows_through_beta(self, vol_pair):
        """Beta parameter should receive a gradient after a forward+backward pass."""
        source, target = vol_pair
        fda = AdaptiveFDA3D(beta_init=0.1, mode="learnable")
        out = fda(source, target)
        loss = out.mean()
        loss.backward()
        assert fda.beta_logit.grad is not None
        assert fda.beta_logit.grad.abs().item() > 0.0

    def test_adaptive_gradient_flows_through_net(self, vol_pair):
        """beta_net parameters should receive gradients."""
        source, target = vol_pair
        fda = AdaptiveFDA3D(beta_init=0.1, mode="adaptive")
        out = fda(source, target)
        loss = out.mean()
        loss.backward()
        for name, param in fda.beta_net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batch_size_two(self):
        """Module should handle batch size > 1 without errors."""
        torch.manual_seed(1)
        source = torch.rand(2, 1, 8, 8, 8)
        target = torch.rand(2, 1, 8, 8, 8)
        for mode in ("fixed", "learnable", "adaptive"):
            fda = AdaptiveFDA3D(beta_init=0.1, mode=mode)
            out = fda(source, target)
            assert out.shape == source.shape, f"Shape mismatch for mode={mode}"

    def test_multi_channel(self):
        """Module should work with in_channels > 1."""
        torch.manual_seed(2)
        source = torch.rand(1, 2, 8, 8, 8)
        target = torch.rand(1, 2, 8, 8, 8)
        for mode in ("fixed", "learnable", "adaptive"):
            fda = AdaptiveFDA3D(beta_init=0.1, mode=mode, in_channels=2)
            out = fda(source, target)
            assert out.shape == source.shape


# ---------------------------------------------------------------------------
# Integration: SiameseUNetBaseline with FDA
# ---------------------------------------------------------------------------

class TestSiameseWithFDA:
    @pytest.mark.parametrize("mode", ["fixed", "learnable", "adaptive"])
    def test_forward_with_fda(self, mode):
        """SiameseUNetBaseline with use_fda=True should forward without errors."""
        torch.manual_seed(42)
        inshape = (16, 16, 16)
        model = vxm.nn.SiameseUNetBaseline(
            inshape=inshape,
            enc_nf=[4, 8],
            dec_nf=[8, 4],
            use_fda=True,
            fda_beta_init=0.1,
            fda_mode=mode,
        )
        source = torch.rand(1, 1, *inshape)
        target = torch.rand(1, 1, *inshape)
        displacement, warped = model(source, target, return_warped_source=True)
        assert displacement.shape == (1, 3, *inshape)
        assert warped.shape == source.shape

    def test_forward_without_fda(self):
        """SiameseUNetBaseline without FDA (baseline) should still work."""
        torch.manual_seed(42)
        inshape = (16, 16, 16)
        model = vxm.nn.SiameseUNetBaseline(
            inshape=inshape,
            enc_nf=[4, 8],
            dec_nf=[8, 4],
            use_fda=False,
        )
        source = torch.rand(1, 1, *inshape)
        target = torch.rand(1, 1, *inshape)
        displacement, warped = model(source, target, return_warped_source=True)
        assert displacement.shape == (1, 3, *inshape)

    def test_fda_module_accessible_on_model(self):
        """The fda attribute should be an AdaptiveFDA3D when use_fda=True."""
        model = vxm.nn.SiameseUNetBaseline(
            inshape=(16, 16, 16),
            enc_nf=[4, 8],
            dec_nf=[8, 4],
            use_fda=True,
        )
        assert isinstance(model.fda, AdaptiveFDA3D)

    def test_no_fda_attribute_is_none_when_disabled(self):
        model = vxm.nn.SiameseUNetBaseline(
            inshape=(16, 16, 16),
            enc_nf=[4, 8],
            dec_nf=[8, 4],
            use_fda=False,
        )
        assert model.fda is None
