"""Arm C: rasterize each protein as 4 Gaussians onto a 2D grid, encode with a CNN.

Design notes that are load-bearing:
  * 4 primitives per protein (centre + 3 blades), not 1. A single Gaussian is
    180-symmetric and cannot distinguish theta from theta+60 -- it would destroy the
    3-fold orientation this work exists to add.
  * 2D CNN with z as CHANNELS, not a 3D CNN. A 3D conv encodes z-translation
    equivariance, a symmetry this system does not have: the bilayer midplane is an
    absolute z reference and proteins are embedded in it. Measured z extent is
    429.7 A against a 3440.4 A lateral box (8:1), so a 3D grid also costs 9x the
    voxels at the existing 50 A mesh spacing (18x at 25 A) for a wrong prior.
  * Circular padding: in-plane the system IS translation-invariant under PBC.
"""
import torch
import torch.nn as nn


class SplatEncoder(nn.Module):
    def __init__(self, out_dim, n_particles=7, box_size=856.1959882301632,
                 grid=69, z_slabs=5, sigma=8.0, hidden=32):
        super().__init__()
        self.P, self.G, self.S = int(n_particles), int(grid), int(z_slabs)
        self.box, self.sigma = float(box_size), float(sigma)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.S, hidden, 3, padding=1, padding_mode='circular'), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, padding_mode='circular'), nn.SiLU(),
        )
        self.head = nn.Linear(hidden, out_dim)

    def rasterize(self, centers_flat, blade):
        """-> [B, z_slabs, G, G] density. Analytic Gaussian on the lattice; 28
        primitives on a 69x69 grid is cheap enough not to need a CUDA rasterizer."""
        B = centers_flat.shape[0]
        c = centers_flat.reshape(B, self.P, 3)
        pts = torch.cat([c.unsqueeze(2), c.unsqueeze(2) + blade], dim=2)   # [B,P,4,3]
        pts = pts.reshape(B, self.P * 4, 3)
        xy = torch.remainder(pts[..., :2], self.box)                       # [B,M,2]
        z = pts[..., 2]

        ax = (torch.arange(self.G, device=pts.device, dtype=pts.dtype) + 0.5) * (self.box / self.G)
        dx = xy[..., 0:1] - ax.view(1, 1, -1)                              # [B,M,G]
        dy = xy[..., 1:2] - ax.view(1, 1, -1)
        dx = dx - self.box * torch.round(dx / self.box)                    # PBC
        dy = dy - self.box * torch.round(dy / self.box)
        gx = torch.exp(-0.5 * (dx / self.sigma) ** 2)
        gy = torch.exp(-0.5 * (dy / self.sigma) ** 2)

        # z -> soft slab membership, so tilt shows up as differences BETWEEN channels
        zc = torch.linspace(-1.0, 1.0, self.S, device=pts.device, dtype=pts.dtype)
        zn = torch.tanh(z / (0.25 * self.box)).unsqueeze(-1)               # [B,M,1]
        # softmax kernel width is tied to slab spacing (2/(S-1) in the [-1,1] zc
        # range), not a free constant -- at the default S=5 this reduces to the
        # original 0.08, but it now scales with z_slabs so sweeping that knob
        # actually buys finer z resolution instead of silently blurring slabs
        # together (or, at large S, collapsing the softmax to near-uniform).
        slab_spacing = 2.0 / max(self.S - 1, 1)
        denom = 0.32 * slab_spacing ** 2
        wz = torch.softmax(-((zn - zc.view(1, 1, -1)) ** 2) / denom, dim=-1)  # [B,M,S]

        # [B,M,S] x [B,M,G] x [B,M,G] -> [B,S,G,G]
        return torch.einsum('bms,bmi,bmj->bsij', wz, gx, gy)

    def forward(self, centers_flat, blade):
        h = self.cnn(self.rasterize(centers_flat, blade))
        return self.head(h.mean(dim=(-2, -1)))
