"""
ReVive: A minimal corrupt-and-repair denoising network for
financial time series anomaly detection.

This is an independent reimplementation of the architecture described in:
  Ozer, Ebeling, Enzweiler, "Just Repair: A Minimal Denoising Network for
  Time Series Anomaly Detection" (arXiv:2604.17388, 2026)

No official source code was available at the time of writing (repository
contained only a README, marked "code coming soon upon paper acceptance").
This implementation was written from scratch based solely on the
architectural description and equations in the published paper, and is
independently named to distinguish it from the original authors' work.

Architecture (paper Section III-B, Fig. 2):
    Input x (B, W, C)
      -> Conv1D 1x1 projection: C -> H
      -> Residual Block:
            DW-Conv (kernel=5, depthwise, per-channel temporal filtering)
            PW-Conv (1x1, cross-channel mixing)
            GELU
            + residual connection
      -> Conv1D 1x1 projection: H -> C   (zero-initialized)
      -> Global skip: f(x) = x + delta(x)

At initialization, the zero-init output projection makes f(x) = x exactly,
so the network starts as the identity and learns corrections from there.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableResidualBlock(nn.Module):
    """
    A single depthwise-separable convolutional residual block.

    Depthwise conv (kernel_size, groups=H) does per-channel temporal
    filtering -- this is where the network learns to smooth out the
    Gaussian noise corruption applied during training.

    Pointwise conv (1x1) mixes information across channels -- this is
    where the network learns to reconstruct masked/dropped channels
    from the surviving ones.
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.dw_conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=kernel_size,
            padding=padding, groups=hidden_dim, bias=True,
        )
        self.pw_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, H, W)
        out = self.dw_conv(h)
        out = self.pw_conv(out)
        out = self.act(out)
        return h + out  # residual connection inside the block


class ReVive(nn.Module):
    """
    Full repair network: input projection -> one residual block ->
    zero-initialized output projection -> global skip connection.

    Args:
        num_channels: number of input channels/features (C)
        hidden_dim: hidden width (H) of the residual block. Paper default: 128.
        kernel_size: depthwise conv kernel size. Paper default: 5.
    """

    def __init__(self, num_channels: int, hidden_dim: int = 128, kernel_size: int = 5):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        self.in_proj = nn.Conv1d(num_channels, hidden_dim, kernel_size=1)
        self.block = DepthwiseSeparableResidualBlock(hidden_dim, kernel_size)
        self.out_proj = nn.Conv1d(hidden_dim, num_channels, kernel_size=1)

        # Zero-initialize the output projection so f(x) = x at init.
        # This means training starts from the identity function and only
        # has to learn a *correction*, not a full reconstruction from scratch.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W, C) -- batch, window length, channels
        returns: (B, W, C) -- repaired/reconstructed window
        """
        h = x.transpose(1, 2)          # -> (B, C, W) for Conv1d
        h = self.in_proj(h)            # -> (B, H, W)
        h = self.block(h)              # -> (B, H, W)
        delta = self.out_proj(h)       # -> (B, C, W)
        delta = delta.transpose(1, 2)  # -> (B, W, C)
        return x + delta               # global residual: f(x) = x + delta(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Quick sanity check: forward pass shape + parameter count.
    # For C=1, H=128, the paper reports 17,665 parameters -- use this
    # to confirm the implementation matches the spec before training.
    model = ReVive(num_channels=1, hidden_dim=128, kernel_size=5)
    dummy = torch.randn(4, 100, 1)  # batch=4, W=100, C=1
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameter count: {model.num_parameters()} (paper reports 17,665 at C=1)")
    # At init, output should equal input exactly (zero-init check)
    print(f"Zero-init check (should be ~0): {(out - dummy).abs().max().item():.8f}")
