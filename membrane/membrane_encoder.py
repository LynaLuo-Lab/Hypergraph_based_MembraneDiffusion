"""Membrane patch sampler + encoder for the position-sampled membrane model.

At each rollout step the model samples a fixed-radius disc of the bilayer height
field at its PREDICTED (detached) particle positions, from the current frame's
full mesh grid, then encodes the 7 discs into a single context vector that
REPLACES the geometry embedding in the decoder's cross-attention.

Registration (verified 2026-06-14): model coords are SCALED [0,856.196];
real-A = scaled * (BOX/856.196); mesh grid is 69x69 @ 50A, origin -1720.18,
periodic with grid-period NG*DG.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_discs(mesh, pos_scaled, offsets, G0, DG, NG, scale):
    """Sample per-protein discs from one frame's mesh grid (PBC-bilinear).

    mesh        : [B, 2, NG, NG]  leaflet height grids (real-A)
    pos_scaled  : [B, P, 2]       scaled xy of each protein (detached upstream)
    offsets     : [K]             disc grid offsets in real-A (e.g. -400..400 step 50)
    returns     : [B, P, 2, K, K] discs (raw heights)
    """
    B, C, _, _ = mesh.shape
    P = pos_scaled.shape[1]
    K = offsets.shape[0]
    real = pos_scaled * scale                                   # [B,P,2]
    OX, OY = torch.meshgrid(offsets, offsets, indexing='ij')    # [K,K]
    qx = real[:, :, 0, None, None] + OX                         # [B,P,K,K]
    qy = real[:, :, 1, None, None] + OY
    fx = ((qx - G0) / DG) % NG
    fy = ((qy - G0) / DG) % NG
    i0 = torch.floor(fx).long() % NG; i1 = (i0 + 1) % NG; tx = fx - torch.floor(fx)
    j0 = torch.floor(fy).long() % NG; j1 = (j0 + 1) % NG; ty = fy - torch.floor(fy)
    mf = mesh.reshape(B, C, NG * NG)

    def gather(ii, jj):
        idx = (ii * NG + jj).reshape(B, 1, -1).expand(B, C, -1)  # [B,C,P*K*K]
        return torch.gather(mf, 2, idx).reshape(B, C, P, K, K)

    v00 = gather(i0, j0); v01 = gather(i0, j1)
    v10 = gather(i1, j0); v11 = gather(i1, j1)
    txc = tx.unsqueeze(1); tyc = ty.unsqueeze(1)                 # [B,1,P,K,K]
    out = (v00 * (1 - txc) * (1 - tyc) + v10 * txc * (1 - tyc) +
           v01 * (1 - txc) * tyc + v11 * txc * tyc)              # [B,C,P,K,K]
    return out.permute(0, 2, 1, 3, 4).contiguous()              # [B,P,C,K,K]


class MembranePatchEncoder(nn.Module):
    """7 discs -> single membrane context vector (drop-in for geo curr_emb in the
    decoder cross-attention). Sampling + height-centering + circular mask + CNN."""

    def __init__(self, out_dim, n_particles=7, radius=400.0, spacing=50.0,
                 G0=-1720.18, NG=69, box=3440.36, scaled_box=856.196, d_disc=32):
        super().__init__()
        self.P = n_particles
        self.G0, self.DG, self.NG = G0, spacing, NG
        self.scale = box / scaled_box
        K = int(radius // spacing)
        offs = torch.arange(-K, K + 1).float() * spacing
        OX, OY = torch.meshgrid(offs, offs, indexing='ij')
        mask = (torch.sqrt(OX ** 2 + OY ** 2) <= radius).float()
        self.register_buffer('offsets', offs, persistent=False)
        self.register_buffer('mask', mask.view(1, 1, *mask.shape), persistent=False)
        self.center = len(offs) // 2
        # per-disc CNN (2 leaflet channels -> d_disc)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, d_disc, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1))                              # [.,d_disc,1,1]
        self.combine = nn.Sequential(
            nn.LayerNorm(n_particles * d_disc),
            nn.Linear(n_particles * d_disc, out_dim), nn.SiLU())
        self.d_disc = d_disc

    def forward(self, mesh_frame, pos_scaled):
        """mesh_frame [B,2,NG,NG], pos_scaled [B,P,2] -> context [B,out_dim]."""
        discs = sample_discs(mesh_frame, pos_scaled, self.offsets,
                             self.G0, self.DG, self.NG, self.scale)   # [B,P,2,K,K]
        # centre each disc on the protein's local height, then circular mask
        ctr = discs[..., self.center, self.center].unsqueeze(-1).unsqueeze(-1)
        discs = (discs - ctr) * self.mask
        B, P, C, K, _ = discs.shape
        feat = self.cnn(discs.reshape(B * P, C, K, K)).reshape(B, P * self.d_disc)
        return self.combine(feat)                                    # [B,out_dim]
