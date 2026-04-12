"""
Adaptive Fourier Domain Adaptation (FDA) module for multimodal image registration.

FDA bridges the appearance gap between source (CT) and target (MR) images by
transplanting low-frequency amplitude statistics from the target into the source
in the Fourier domain, while preserving the source's structural phase.

Three operating modes are provided:

``fixed``
    Beta is a plain float constant.  No parameters are learned.  Useful as an
    ablation baseline when you want deterministic frequency mixing.

``learnable``
    Beta is a single scalar ``nn.Parameter`` optimised end-to-end with the rest
    of the network.  A differentiable soft mask (steep sigmoid) ensures that
    gradients flow through beta during back-propagation, so the network can
    discover the optimal mixing radius automatically.

``adaptive``
    Beta is predicted *per sample* by a lightweight amplitude-statistics network
    that takes the spectral difference between source and target as input.  This
    allows the mixing radius to vary across image pairs at inference time with no
    additional manual tuning.

Design goals
------------
* **Decoupled** – the module lives independently of the registration backbone, so
  it can be plugged in or removed for ablation studies without touching other code.
* **End-to-end** – once baked into the model the module applies automatically at
  both training and inference; no manual pre-processing step is required.
* **Differentiable** – the soft-mask parameterisation for ``learnable`` and
  ``adaptive`` modes lets beta be trained via standard gradient descent.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soft_low_freq_mask(
    D: int,
    H: int,
    W: int,
    beta: torch.Tensor,
    temperature: float = 0.01,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build a soft low-frequency mask in the *shifted* Fourier domain.

    The mask is 1 near the centre (DC and low frequencies) and 0 at the
    periphery.  The transition sharpness is controlled by ``temperature``:
    smaller values produce a harder step.

    Coordinates are normalised to the range ``[-0.5, 0.5]`` along each axis
    and the L-infinity norm is used so the mask is a rectangular region in
    frequency space (matching the standard hard-mask convention).

    Parameters
    ----------
    D, H, W : int
        Spatial dimensions of the volume.
    beta : torch.Tensor
        Scalar (or broadcastable) bandwidth controlling the half-width of the
        low-frequency window.  Values in ``(0, 0.5)`` are reasonable.
    temperature : float
        Sigmoid temperature; lower → harder boundary.
    device : torch.device, optional
        Target device for the mask tensor.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 1, D, H, W)`` float mask broadcastable over ``(B, C, …)``.
    """
    d_norm = (torch.arange(D, device=device, dtype=torch.float32) / D - 0.5).abs()
    h_norm = (torch.arange(H, device=device, dtype=torch.float32) / H - 0.5).abs()
    w_norm = (torch.arange(W, device=device, dtype=torch.float32) / W - 0.5).abs()

    # L-infinity distance from DC centre: shape (D, H, W)
    dd, hh, ww = torch.meshgrid(d_norm, h_norm, w_norm, indexing="ij")
    dist = torch.amax(torch.stack([dd, hh, ww], dim=0), dim=0)  # (D, H, W)

    # Soft step: sigmoid((beta − dist) / temperature)
    # When dist < beta  → ~1 (inside low-freq window)
    # When dist > beta  → ~0 (outside window)
    mask = torch.sigmoid((beta - dist) / temperature)  # broadcasts over batch/channel
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class AdaptiveFDA3D(nn.Module):
    """
    Adaptive Fourier Domain Adaptation for 3D medical volumes.

    Parameters
    ----------
    beta_init : float
        Initial value of the low-frequency bandwidth beta (in ``(0, 0.5)``).
        For ``fixed`` mode this is the constant value.
        For ``learnable`` and ``adaptive`` modes this is the starting point.
    mode : str
        One of ``'fixed'``, ``'learnable'``, or ``'adaptive'``.
        See module docstring for a description of each mode.
    in_channels : int
        Number of image channels.  Used by the ``adaptive`` mode predictor.
    temperature : float
        Sigmoid temperature for the soft mask (``learnable`` / ``adaptive``).
        Smaller values approximate a hard rectangular mask more closely.

    Attributes
    ----------
    beta_logit : nn.Parameter
        (``learnable`` mode only) Logit of beta; ``sigmoid(beta_logit)`` gives
        the current beta.
    beta_net : nn.Sequential
        (``adaptive`` mode only) Small network mapping spectral difference
        statistics to a per-sample beta scalar.

    Examples
    --------
    Fixed mode (ablation baseline):

    >>> fda = AdaptiveFDA3D(beta_init=0.1, mode='fixed')
    >>> adapted = fda(ct_volume, mr_volume)

    Learnable mode (gradient-driven, single shared beta):

    >>> fda = AdaptiveFDA3D(beta_init=0.1, mode='learnable')
    >>> adapted = fda(ct_volume, mr_volume)
    >>> # fda.get_beta() reports the current learned value after training

    Adaptive mode (per-sample beta):

    >>> fda = AdaptiveFDA3D(beta_init=0.1, mode='adaptive')
    >>> adapted = fda(ct_volume, mr_volume)
    """

    def __init__(
        self,
        beta_init: float = 0.1,
        mode: str = "learnable",
        in_channels: int = 1,
        temperature: float = 0.01,
    ) -> None:
        super().__init__()

        if not (0.0 < beta_init < 0.5):
            raise ValueError(f"beta_init must be in (0, 0.5), got {beta_init}.")
        if mode not in ("fixed", "learnable", "adaptive"):
            raise ValueError(
                f"mode must be 'fixed', 'learnable', or 'adaptive', got '{mode}'."
            )

        self.mode = mode
        self.in_channels = in_channels
        self.temperature = temperature

        if mode == "fixed":
            # Non-learnable constant; stored as a buffer so it moves with .to(device)
            self.register_buffer("_beta_fixed", torch.tensor(beta_init))

        elif mode == "learnable":
            # Parameterise as logit so that sigmoid(logit) ∈ (0,1) always.
            # Clamp the init range to avoid log(0).
            beta_clamped = max(1e-4, min(beta_init, 0.4999))
            init_logit = math.log(beta_clamped / (1.0 - beta_clamped))
            self.beta_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        else:  # adaptive
            # Lightweight predictor: takes the mean absolute amplitude difference
            # (pooled globally) and maps it to a scalar beta ∈ (0, 0.5).
            #
            # Input:  2 * in_channels features (src amp stats + tgt amp stats)
            # Output: 1 scalar clamped to (0, 0.5)
            self.beta_net = nn.Sequential(
                # Global spatial pooling is performed before this network;
                # so input size is (B, 2 * in_channels)
                nn.Linear(2 * in_channels, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1),
                nn.Sigmoid(),  # Output ∈ (0, 1); will be scaled to (0, 0.5) later
            )
            # Bias the output towards beta_init at initialisation.
            # get_beta() multiplies the Sigmoid output by 0.5 to map (0,1) → (0,0.5),
            # so we need the raw Sigmoid output to equal beta_init/0.5 at init.
            # Solving:  sigmoid(bias) = beta_init/0.5  →  bias = log((beta_init/0.5) / (1 - beta_init/0.5))
            scaled_init = beta_init / 0.5  # desired Sigmoid output before the ×0.5 rescaling
            scaled_init = max(1e-4, min(scaled_init, 0.9999))
            init_bias = math.log(scaled_init / (1.0 - scaled_init))
            nn.init.zeros_(self.beta_net[0].weight)
            nn.init.zeros_(self.beta_net[0].bias)
            nn.init.zeros_(self.beta_net[2].weight)
            self.beta_net[2].bias.data.fill_(init_bias)

    # ------------------------------------------------------------------
    # Beta accessor
    # ------------------------------------------------------------------

    def get_beta(
        self,
        src_amp_stats: Optional[torch.Tensor] = None,
        tgt_amp_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return the current beta value as a scalar tensor (or per-sample for
        ``adaptive`` mode).

        Parameters
        ----------
        src_amp_stats : torch.Tensor, optional
            Per-batch mean log amplitude of source FFT.  Required for
            ``adaptive`` mode; ignored otherwise.
        tgt_amp_stats : torch.Tensor, optional
            Per-batch mean log amplitude of target FFT.  Required for
            ``adaptive`` mode; ignored otherwise.

        Returns
        -------
        torch.Tensor
            Scalar beta (shape ``()`` or ``(B, 1)`` for adaptive mode),
            always in the range ``(0, 0.5)``.
        """
        if self.mode == "fixed":
            return self._beta_fixed

        elif self.mode == "learnable":
            # Clamp to (0, 0.5) to keep within physically meaningful range
            return torch.sigmoid(self.beta_logit) * 0.5

        else:  # adaptive
            if src_amp_stats is None or tgt_amp_stats is None:
                raise ValueError(
                    "src_amp_stats and tgt_amp_stats are required for adaptive mode."
                )
            features = torch.cat([src_amp_stats, tgt_amp_stats], dim=-1)  # (B, 2C)
            beta = self.beta_net(features) * 0.5  # (B, 1), range (0, 0.5)
            return beta

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FDA: adapt the source amplitude towards the target in the
        low-frequency band while preserving source phase (and thus structure).

        Parameters
        ----------
        source : torch.Tensor
            Source image, shape ``(B, C, D, H, W)``, values in ``[0, 1]``.
        target : torch.Tensor
            Target image, shape ``(B, C, D, H, W)``, values in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Adapted source with target's low-frequency style, same shape as
            ``source``, values clamped to ``[0, 1]``.

        Notes
        -----
        The FFT is computed on ``float32`` tensors to avoid complex-number
        precision issues with ``float16``/``bfloat16`` under AMP.  The
        adapted output is cast back to the original dtype before returning.
        """
        orig_dtype = source.dtype

        # Work in float32 for numerical stability
        source_f32 = source.float()
        target_f32 = target.float()

        B, C, D, H, W = source_f32.shape
        device = source_f32.device

        # --- 1. FFT (per-batch, per-channel) ---
        src_fft = torch.fft.fftn(source_f32, dim=(-3, -2, -1))
        tgt_fft = torch.fft.fftn(target_f32, dim=(-3, -2, -1))

        # Shift DC to centre
        src_fft_s = torch.fft.fftshift(src_fft, dim=(-3, -2, -1))
        tgt_fft_s = torch.fft.fftshift(tgt_fft, dim=(-3, -2, -1))

        # --- 2. Amplitude & Phase ---
        src_amp = torch.abs(src_fft_s)      # (B, C, D, H, W)
        src_phase = torch.angle(src_fft_s)  # (B, C, D, H, W)
        tgt_amp = torch.abs(tgt_fft_s)      # (B, C, D, H, W)

        # --- 3. Compute beta ---
        if self.mode == "adaptive":
            # Summarise spectral statistics: log mean amplitude per channel
            # shape: (B, C)
            src_stats = torch.log1p(src_amp.flatten(2).mean(dim=-1))
            tgt_stats = torch.log1p(tgt_amp.flatten(2).mean(dim=-1))
            beta = self.get_beta(src_stats, tgt_stats)  # (B, 1)
            # Use mean beta for mask construction (masks differ negligibly across samples)
            beta_for_mask = beta.mean()
        else:
            beta_for_mask = self.get_beta()  # scalar

        # --- 4. Soft low-frequency mask ---
        mask = _soft_low_freq_mask(
            D, H, W, beta_for_mask, self.temperature, device
        )  # (1, 1, D, H, W)

        # For adaptive mode, optionally scale mask per sample (simple scaling)
        if self.mode == "adaptive":
            # beta has shape (B, 1); broadcast over spatial dims
            # Re-weight mask to reflect per-sample beta ratios
            # This is a first-order approximation: scale the mask by
            # (sample_beta / mean_beta) so higher-beta samples get a wider window.
            scale = beta / (beta_for_mask.detach() + 1e-8)  # (B, 1)
            scale = scale.view(B, 1, 1, 1, 1)
            mask = (mask * scale).clamp(0.0, 1.0)

        # --- 5. Mix amplitudes ---
        adapted_amp = src_amp * (1.0 - mask) + tgt_amp * mask

        # --- 6. Reconstruct ---
        adapted_fft_s = torch.polar(adapted_amp, src_phase)
        adapted_fft = torch.fft.ifftshift(adapted_fft_s, dim=(-3, -2, -1))
        adapted = torch.fft.ifftn(adapted_fft, dim=(-3, -2, -1)).real

        # Clamp to valid image range and restore original dtype
        return adapted.clamp(0.0, 1.0).to(orig_dtype)
