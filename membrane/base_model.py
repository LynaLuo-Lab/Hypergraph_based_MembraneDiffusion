"""
BaseBrownianModel v2

- Stochastic head: BrownianHead vs OUHead via config['latent_mode']
- No KL loss
- Decoder: decoder(z, curr_emb) -> delta_sample, delta_mu, delta_sigma (stateless)
- Subclasses set: self.geo_embed, self.encoder, self.decoder
"""
import os, warnings, traceback, math
import lightning as L
import math
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt_util
from nbcore.transition_heads import (
    MLPGaussTransitionHead as LatentTransitionHead,
    squeeze_transition_params,
)
from membrane.vacf_loss import compute_vacf_quantile_loss
from membrane.geometry_features import (BLADE_ICS, GEOM_ARMS, blade_theta,
                                                  rotation_step_angle, tilt_angle)

warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

# =========================================================================
#  GAT Spatial Embedding
# =========================================================================
LOG_FLOOR = 1.0     # a pair closer than this (an overlap) counts as this close in ln(distance); same unit as the distances
DLOG_LAGS = (1, 5, 10, 25, 50, 100, 200, 300, 500, 1000, 2000, 3000)    # the cohesion loss's lags


def _ln_dist(pw, floor):
    return torch.log((pw if pw.dtype == torch.float64 else pw.float()).clamp_min(floor))


def _gauss_kl_parts(t, p):
    """KL(N_t || N_p) between Gaussians fitted to the target and predicted samples, and its mean and spread parts,
    signed + when the prediction's mean is larger and its spread wider"""
    mt, st = t.mean(), t.std(correction=0)
    mp, sp = p.mean(), p.std(correction=0)
    mean = (mt - mp) ** 2 / (2 * sp ** 2)
    spread = torch.log(sp / st) + st ** 2 / (2 * sp ** 2) - 0.5
    return mean + spread, torch.sign(mp - mt) * mean, torch.sign(sp - st) * spread


def logpair_kl(pred_pw, tgt_pw, floor=LOG_FLOOR):
    """[B, T, pairs] x,y pair distances -> KL(target || prediction) between Gaussians fitted to ln(distance), pooled
    over the batch, with its mean part (+ proteins further apart) and spread part (+ distances more varied). Two MD
    trajectories of one system score close to each other and far from random placement or another system. The KL does
    not depend on the distances' unit; floor is in that unit."""
    return _gauss_kl_parts(_ln_dist(tgt_pw, floor), _ln_dist(pred_pw, floor))


def dlogpair_kl(pred_pw, tgt_pw, lags=DLOG_LAGS, floor=LOG_FLOOR):
    """the same on the signed change in ln(distance) over each lag that fits the window, averaged over lags: mean
    part + when pairs drift apart, spread part + when pair distances fluctuate more"""
    lp, lt = _ln_dist(pred_pw, floor), _ln_dist(tgt_pw, floor)
    T = min(lp.size(1), lt.size(1))
    parts = [_gauss_kl_parts(lt[:, L:T] - lt[:, :T - L], lp[:, L:T] - lp[:, :T - L]) for L in lags if L < T]
    if not parts:
        z = lp.new_zeros(())
        return z, z, z
    return tuple(torch.stack(v).mean() for v in zip(*parts))


def pair_dir_features(xyz, box_size, shells):
    """[..., P, 3] -> [..., P, 2*shells]: per protein and distance shell, the summed x,y unit
    vectors towards the neighbours in that shell, laid out [shell, (x, y)]. The pair
    distances say how far a neighbour is; this is the only input that says which way.
    Shells are hats centred at m*box/(2*(shells+1)), m = 1..shells: a neighbour is split
    between the two nearest, and the weight is 0 at contact and from half a box on, where
    the minimum-image direction flips."""
    r = xyz[..., None, :, :2] - xyz[..., :, None, :2]                 # [..., i, j, 2] = x_j - x_i
    r = r - box_size*torch.round(r/box_size)
    d = torch.sqrt((r**2).sum(-1, keepdim=True) + 1e-8)
    m = torch.arange(1, shells + 1, device=xyz.device, dtype=xyz.dtype)
    k = (1 - (d*(2*(shells + 1)/box_size) - m).abs()).clamp_min(0)    # [..., i, j, shells]
    return (k.unsqueeze(-1)*(r/d).unsqueeze(-2)).sum(-3).flatten(-2)


def ic_once_node_features(xyz, prev, box_size, keep_z, pair_shells=0):
    """[..., P, 3] -> [..., P, 9] = [sin | cos of absolute position | min-image step].
    prev=None is the IC frame: absolute position, zero step. Every later frame gets
    the step from prev and no absolute position -- except z when keep_z, because the
    bilayer fixes z in a way it does not fix x or y. Shared by the embedding and the
    tokenized decoder's position keys so the two cannot drift apart.
    pair_shells > 0 appends pair_dir_features on every frame: 9 + 2*pair_shells wide."""
    sc = 2*math.pi*torch.remainder(xyz, box_size)/box_size
    pos = torch.cat([torch.sin(sc), torch.cos(sc)], dim=-1)
    if prev is None:
        f = torch.cat([pos, torch.zeros_like(xyz)], dim=-1)
    else:
        dx = xyz - prev
        dx = dx - box_size*torch.round(dx/box_size)
        keep = pos.new_tensor([0., 0., 1., 0., 0., 1.]) if keep_z else pos.new_zeros(6)
        f = torch.cat([pos*keep, dx], dim=-1)
    if pair_shells:
        f = torch.cat([f, pair_dir_features(xyz, box_size, pair_shells)], dim=-1)
    return f


class GAT_GeometryEmbedding(nn.Module):
    def __init__(self, box_size=1.0, hidden_dim=32, num_heads=4, n_particles=7,
                 ic_once=False, ic_keep_z=False, pair_dir_shells=0):
        super().__init__()
        if pair_dir_shells and not ic_once:
            raise ValueError("pair_dir_shells is built into the geo_ic_once node features only")
        self.ic_once = ic_once; self.ic_keep_z = ic_keep_z; self.pair_dir_shells = pair_dir_shells
        self.box_size = box_size; self.hidden_dim = hidden_dim
        self.node_input_dim = (9 if ic_once else 6) + 2*pair_dir_shells
        self.n_particles = n_particles
        self.n_coords = n_particles * 3
        self.n_pairs = n_particles * (n_particles - 1) // 2
        self.input_proj = nn.Linear(self.node_input_dim, hidden_dim)
        self.gat_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.GELU(), nn.Linear(hidden_dim*2, hidden_dim))
        r, c = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer('triu_rows', r); self.register_buffer('triu_cols', c)
        self.output_dim = n_particles*hidden_dim + self.n_pairs

    def forward(self, x, active_mask=None, edge_src=None, edge_dst=None, edge_mask=None, prev=None):
        # GAT ignores mask args (accepted for API compatibility with GNN_GeometryEmbedding)
        if x.dim() == 2: x = x.unsqueeze(1)
        B, S, _ = x.size()
        xyz = x[:,:,:self.n_coords].reshape(B,S,self.n_particles,3)
        if self.ic_once:
            if prev is not None:
                if prev.dim() == 2: prev = prev.unsqueeze(1)
                prev = prev[:,:,:self.n_coords].reshape(B,S,self.n_particles,3)
            emb = ic_once_node_features(xyz, prev, self.box_size, self.ic_keep_z, self.pair_dir_shells)
        else:
            sc = 2*math.pi*torch.remainder(xyz, self.box_size)/self.box_size
            emb = torch.cat([torch.sin(sc), torch.cos(sc)], dim=-1)
        n = self.input_proj(emb.view(-1,self.n_particles,self.node_input_dim))
        a, _ = self.gat_layer(n,n,n); n = self.norm1(n+a); n = self.norm2(n+self.mlp(n))
        gat = n.view(B,S,-1)
        d = xyz.unsqueeze(3)-xyz.unsqueeze(2)
        d = d - self.box_size*torch.round(d/self.box_size)
        dm = torch.sqrt((d**2).sum(-1)+1e-8)
        pw = dm[:,:,self.triu_rows,self.triu_cols]/self.box_size
        return torch.cat([gat,pw],dim=-1).contiguous(), dm[:,:,self.triu_rows,self.triu_cols]*self.box_size

    def compute_distances_xy_only(self, x, active_mask=None):
        # active_mask accepted for API compatibility with GNN_GeometryEmbedding
        if x.dim() == 2: x = x.unsqueeze(1)
        B, S, _ = x.size()
        xy = x[:,:,:self.n_coords].reshape(B,S,self.n_particles,3)[:,:,:,:2]
        d = xy.unsqueeze(3)-xy.unsqueeze(2)
        d = d - self.box_size*torch.round(d/self.box_size)
        dm = torch.sqrt((d**2).sum(-1)+1e-8)
        return dm[:,:,self.triu_rows,self.triu_cols]/self.box_size


class GNN_GeometryEmbedding(nn.Module):
    """Message-passing GNN that uses minimum-image displacement vectors as edge features.

    Unlike GAT (content-based attention, geometry only via node encoding),
    this explicitly passes pairwise displacement vectors through the message
    function so spatial structure directly informs inter-particle communication.
    """

    def __init__(self, box_size=1.0, hidden_dim=32, num_heads=4, n_particles=7):
        super().__init__()
        self.box_size = box_size
        self.hidden_dim = hidden_dim
        self.n_particles = n_particles
        self.n_coords = n_particles * 3
        self.n_pairs = n_particles * (n_particles - 1) // 2

        self.input_proj = nn.Linear(6, hidden_dim)

        # Edge feature: 4 = (dx, dy, dz, ||d||), all normalized by box_size
        edge_dim = 4
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        r, c = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer('triu_rows', r)
        self.register_buffer('triu_cols', c)
        # Full directed edge indices for message passing (i != j)
        src, dst = [], []
        for i in range(n_particles):
            for j in range(n_particles):
                if i != j:
                    src.append(i)
                    dst.append(j)
        self.register_buffer('edge_src', torch.tensor(src, dtype=torch.long))
        self.register_buffer('edge_dst', torch.tensor(dst, dtype=torch.long))
        self.output_dim = n_particles * hidden_dim + self.n_pairs

    def forward(self, x, active_mask=None, edge_src=None, edge_dst=None, edge_mask=None):
        """Forward pass with optional multi-system masking.

        Parameters
        ----------
        x : Tensor [B, S, n_particles*3] or [B, n_particles*3]
        active_mask : BoolTensor [B, n_particles], optional
            True for slots with real particles; inactive slots are zeroed.
        edge_src : LongTensor [B, max_edges], optional
            Source node indices for each batch element (padded).
        edge_dst : LongTensor [B, max_edges], optional
            Destination node indices for each batch element (padded).
        edge_mask : BoolTensor [B, max_edges], optional
            True for real edges, False for padding.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, S, _ = x.size()
        xyz = x[:, :, :self.n_coords].reshape(B, S, self.n_particles, 3)

        # Node features: sin/cos of periodic coords
        sc = 2 * math.pi * torch.remainder(xyz, self.box_size) / self.box_size
        emb = torch.cat([torch.sin(sc), torch.cos(sc)], dim=-1)  # [B,S,P,6]
        flat = emb.view(-1, self.n_particles, 6)  # [BS, P, 6]
        n = self.input_proj(flat)  # [BS, P, H]
        BS = n.shape[0]

        # Zero node features for inactive slots (pre-message-pass)
        if active_mask is not None:
            # active_mask: [B, P] -> [B, 1, P, 1] -> broadcast over S and H
            node_alive = active_mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, P, 1]
            n = n.view(B, S, self.n_particles, self.hidden_dim) * node_alive
            n = n.view(BS, self.n_particles, self.hidden_dim)

        # Pairwise minimum-image displacements
        xyz_flat = xyz.reshape(-1, self.n_particles, 3)  # [BS, P, 3]
        d_all = xyz_flat.unsqueeze(2) - xyz_flat.unsqueeze(1)  # [BS, P, P, 3]
        d_all = d_all - self.box_size * torch.round(d_all / self.box_size)
        dm_all = torch.sqrt((d_all ** 2).sum(-1) + 1e-8)  # [BS, P, P]

        if edge_src is not None:
            # External per-sample edge lists: [B, max_edges] -> tile for S -> [BS, max_edges]
            n_edges = edge_src.shape[1]
            e_src = edge_src.unsqueeze(1).expand(B, S, -1).reshape(BS, -1)  # [BS, E]
            e_dst = edge_dst.unsqueeze(1).expand(B, S, -1).reshape(BS, -1)  # [BS, E]
            # edge_mask: [B, max_edges] -> tile -> [BS, max_edges]
            e_mask = edge_mask.unsqueeze(1).expand(B, S, -1).reshape(BS, -1)  # [BS, E]

            # Advanced indexing: gather edge features from d_all [BS, P, P, 3]
            bs_idx = torch.arange(BS, device=n.device).unsqueeze(1).expand(-1, n_edges)  # [BS, E]
            d_edge = d_all[bs_idx, e_src, e_dst, :]   # [BS, E, 3]
            dm_edge = dm_all[bs_idx, e_src, e_dst].unsqueeze(-1)  # [BS, E, 1]
            edge_feat = torch.cat([d_edge / self.box_size, dm_edge / self.box_size], dim=-1)  # [BS, E, 4]

            # Gather node features at source/dest
            h_src = n[bs_idx, e_src, :]  # [BS, E, H]
            h_dst = n[bs_idx, e_dst, :]  # [BS, E, H]
            msg_input = torch.cat([h_src, h_dst, edge_feat], dim=-1)  # [BS, E, 2H+4]
            messages = self.message_mlp(msg_input)  # [BS, E, H]

            # Mask out padded edges
            messages = messages * e_mask.unsqueeze(-1).float()

            # Aggregate: sum messages per destination node
            agg = torch.zeros(BS, self.n_particles, self.hidden_dim,
                              device=n.device, dtype=n.dtype)
            agg.scatter_add_(1, e_dst.unsqueeze(-1).expand(-1, -1, self.hidden_dim), messages)
        else:
            # Standard path: use fixed all-to-all edges
            d_edge = d_all[:, self.edge_src, self.edge_dst, :]  # [BS, E, 3]
            dm_edge = dm_all[:, self.edge_src, self.edge_dst].unsqueeze(-1)  # [BS, E, 1]
            edge_feat = torch.cat([d_edge / self.box_size, dm_edge / self.box_size], dim=-1)  # [BS, E, 4]

            h_src = n[:, self.edge_src, :]  # [BS, E, H]
            h_dst = n[:, self.edge_dst, :]  # [BS, E, H]
            msg_input = torch.cat([h_src, h_dst, edge_feat], dim=-1)  # [BS, E, 2H+4]
            messages = self.message_mlp(msg_input)  # [BS, E, H]

            agg = torch.zeros(BS, self.n_particles, self.hidden_dim,
                              device=n.device, dtype=n.dtype)
            agg.scatter_add_(1, self.edge_dst.unsqueeze(0).unsqueeze(-1).expand(
                BS, -1, self.hidden_dim), messages)

        # Update: residual + norm
        n = self.norm1(n + agg)
        n = self.norm2(n + self.update_mlp(torch.cat([n, agg], dim=-1)))

        # Zero inactive nodes after update
        if active_mask is not None:
            node_alive = active_mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, P, 1]
            n = n.view(B, S, self.n_particles, self.hidden_dim) * node_alive
            n = n.view(BS, self.n_particles, self.hidden_dim)

        gnn = n.view(B, S, -1)
        pw = dm_all.view(-1, self.n_particles, self.n_particles)
        pw = pw[:, self.triu_rows, self.triu_cols] / self.box_size
        pw = pw.view(B, S, -1)
        return torch.cat([gnn, pw], dim=-1).contiguous(), \
            dm_all.view(-1, self.n_particles, self.n_particles)[:, self.triu_rows, self.triu_cols].view(B, S, -1) * self.box_size

    def compute_distances_xy_only(self, x, active_mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, S, _ = x.size()
        xy = x[:, :, :self.n_coords].reshape(B, S, self.n_particles, 3)[:, :, :, :2]
        d = xy.unsqueeze(3) - xy.unsqueeze(2)
        d = d - self.box_size * torch.round(d / self.box_size)
        dm = torch.sqrt((d ** 2).sum(-1) + 1e-8)
        return dm[:, :, self.triu_rows, self.triu_cols] / self.box_size


# =========================================================================
#  LSTM Encoder
# =========================================================================
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=21, noise_rank=16, num_layers=2,
                 dropout=0.0, latent_type='brownian', gmm_components=4):
        super().__init__()
        self.latent_dim=latent_dim; self.hidden_dim=hidden_dim; self.noise_rank=noise_rank; self.num_layers=num_layers
        self.ln_embed = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0)
        for name, p in self.lstm.named_parameters():
            if "bias" in name: n=p.size(0); p.data[n//4:n//2].fill_(1.0)
        self.ln_lstm_out = nn.LayerNorm(hidden_dim)
        self.transition_head = LatentTransitionHead(
            input_dim=hidden_dim,
            latent_dim=latent_dim,
            noise_rank=noise_rank,
            latent_type=latent_type,
            gmm_components=gmm_components,
        )

    def forward(self, embedded_x, state=None):
        x = self.ln_embed(embedded_x)
        out, (h,c) = self.lstm(x, state)
        out = self.ln_lstm_out(out)
        params = self.transition_head(out)
        params = params._replace(mu=10 * torch.tanh(params.mu / 10))
        return params, (h,c)

# =========================================================================
#  Stochastic Heads — Brownian vs OU
# =========================================================================
class StochasticHead(nn.Module):
    def __init__(self, latent_dim, noise_rank, dt):
        super().__init__(); self.latent_dim=latent_dim; self.noise_rank=noise_rank; self.dt=dt
    def _unpack(self, sf):
        L = sf[:,:self.latent_dim*self.noise_rank].reshape(-1,self.latent_dim,self.noise_rank)
        D = F.softplus(sf[:,self.latent_dim*self.noise_rank:]) + 0.1
        return L, D
    def _noise(self, sf, dev):
        L, D = self._unpack(sf)
        return torch.bmm(L, torch.randn(L.size(0),self.noise_rank,1,device=dev)).squeeze(2) + D*torch.randn(D.size(0),self.latent_dim,device=dev)
    def step(self, mu, sf, z_prev): raise NotImplementedError

class BrownianHead(StochasticHead):
    """dz = mu*dt + sigma*dW"""
    def step(self, mu, sf, z_prev):
        o=mu.dtype; mu,sf=mu.float(),sf.float()
        return (z_prev.float() + mu*self.dt + self._noise(sf,mu.device)*(self.dt**0.5)).to(o)

class OUHead(StochasticHead):
    """dz = (-theta*z + mu)*dt + sigma*dW  |  theta=dt -> tau_corr=1 step"""
    def __init__(self, latent_dim, noise_rank, dt, theta=None):
        super().__init__(latent_dim,noise_rank,dt); self.theta = theta if theta is not None else dt
    def step(self, mu, sf, z_prev):
        o=mu.dtype; mu,sf=mu.float(),sf.float(); z=z_prev.float()
        return (z + (-self.theta*z + mu)*self.dt + self._noise(sf,mu.device)*(self.dt**0.5)).to(o)

def make_stochastic_head(config):
    ld,nr,dt = config.get('latent_dim',21), config.get('noise_rank',16), config.get('dt',0.0001)
    mode = config.get('latent_mode','brownian')
    if mode=='brownian': return BrownianHead(ld,nr,dt)
    elif mode=='ou': return OUHead(ld,nr,dt,theta=config.get('ou_theta',dt))
    else: raise ValueError(f"Unknown latent_mode '{mode}'")

# =========================================================================
#  BaseBrownianModel
# =========================================================================
class RotationHead(nn.Module):
    """Per-protein rotational step head: curr_emb -> (mu, sigma) per protein, then
    sample Delta-theta ~ N(mu, sigma) (dof=1) or a full Delta-rotation-vector
    (dof=3, arm B/se3). Decoupled from the position latent (GT rotation is a
    driftless Gaussian with ~0 rotation-translation coupling). Init mu~0, sigma
    ~ the measured per-frame angular step (~1.9deg = 0.033 rad), same init for
    both dof cases."""

    def __init__(self, in_dim, n_particles, sigma_init=0.0332, hidden=64, dof=1):
        super().__init__()
        self.n_particles = n_particles
        self.dof = int(dof)                       # 1 = theta only, 3 = full rotation vector
        out = n_particles * self.dof
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.SiLU())
        self.mu_head = nn.Linear(hidden, out)
        self.logsig_head = nn.Linear(hidden, out)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logsig_head.weight)
        nn.init.constant_(self.logsig_head.bias, float(math.log(max(sigma_init, 1e-4))))

    def forward(self, emb):
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        h = self.net(emb)
        mu = 0.3 * torch.tanh(self.mu_head(h))                 # bounded drift (GT drift ~0)
        sigma = torch.exp(self.logsig_head(h).clamp(-8.0, 1.0))
        out = mu + sigma * torch.randn_like(mu)
        if self.dof == 1:
            return out                                         # [B, n_particles] -- unchanged
        return out.reshape(-1, self.n_particles, self.dof)     # [B, n_particles, 3]


def arm_points_3d(centers_flat, blade):
    """centers_flat [B,P*3], blade [B,P,3,3] -> [B,4P,3] point cloud (centre then the
    3 blades per protein), ordered so point p belongs to protein p//4 -- the same
    layout the 2D arm28_i/j buffers already index. Module-level (not a method) so
    geom_arm='se3' code and tests can call it without a model instance. Points are
    raw offsets, never wrapped (same convention as the 2D arm_points) -- no
    box_size needed here."""
    if blade is None:
        raise ValueError("arm_points_3d requires blade, got None")
    B, P = blade.shape[0], blade.shape[1]
    c = centers_flat.reshape(B, P, 3)
    pts = torch.cat([c.unsqueeze(2), c.unsqueeze(2) + blade], dim=2)   # [B,P,4,3]
    return pts.reshape(B, 4 * P, 3)


_TRIU3D = {}


def _triu_pairs(P, device):
    """Cached upper-triangle pair index for arm_pair_feature_3d, on the operand's
    device. The 2D twin gets these from registered buffers; rebuilding a CPU index
    per call and using it to gather a CUDA tensor forces a host->device copy every
    step -- measured 96.4us/call vs 70.6us cached at the stability rollout's B=512,
    ~1s over its 40k steps."""
    key = (P, str(device))
    idx = _TRIU3D.get(key)
    if idx is None:
        # inference_mode(False) around the build: the first caller is often a
        # predict_future rollout, and an inference tensor cached there would poison
        # every later autograd-tracked forward ("Inference tensors cannot be saved
        # for backward").
        with torch.inference_mode(False):
            idx = torch.triu_indices(P, P, 1).to(device)
        _TRIU3D[key] = idx
    return idx


def arm_pair_feature_3d(centers_flat, blade, box_size):
    """Arm-aware closest approach in full 3D: min inter-body xyz distance per
    protein pair, normalized. Unlike the piezo (non-membrane) fork, THIS model's
    whole point is that proteins sit at different bilayer depths -- the membrane
    module exists to represent that z deformation -- so inter-protein z
    separation is real signal here, not a confound to strip out. A flat trimer at
    the same z as the 2D case reproduces the 2D answer; different centre z or a
    genuine blade tilt both legitimately change this feature, and both should."""
    if blade is None:
        raise ValueError("arm_pair_feature_3d requires blade, got None")
    B, P = blade.shape[0], blade.shape[1]
    four = arm_points_3d(centers_flat, blade).reshape(B, P, 4, 3)
    r, cix = _triu_pairs(P, blade.device)
    pi, pj = four[:, r], four[:, cix]                                  # [B,n_pairs,4,3]
    d = pi.unsqueeze(3) - pj.unsqueeze(2)                              # [B,n_pairs,4,4,3]
    d = d - box_size * torch.round(d / box_size)
    dist = d.norm(dim=-1).reshape(B, r.shape[0], 16)
    return dist.min(dim=-1).values / box_size


def _sched_shape(sch, epoch, horizon, cycle_epochs, cycle_mult):
    """Decreasing schedule shape in [0,1]: 1.0 at epoch 0, 0.0 at `horizon`.

    Shared by ev_force_frac (which uses it as-is) and ev_w_frac (which mirrors it to
    ramp UP), so the two schedules cannot drift apart. Module-level rather than a
    method because the slurm preflights call BaseBrownianModel.ev_force_frac(_S, e)
    UNBOUND against a plain attribute-only stub -- a self.<helper> lookup there would
    AttributeError and take the launcher's fail-fast with it.
    """
    if sch == 'constant':
        return 1.0
    if sch in ('linear', 'cosine'):
        if horizon <= 0:
            return 1.0
        t = min(1.0, max(0.0, epoch / float(horizon)))
        return (1.0 - t) if sch == 'linear' else 0.5 * (1.0 + math.cos(math.pi * t))
    # cosine_restarts: walk cycles until `epoch` lands inside one. The LAST cycle is
    # truncated to end exactly at `horizon`. Without that, T0=5 mult=2 restarts at
    # epoch 35 and runs 40 epochs, so a 40-epoch job ENDS back at 0.98 -- the opposite
    # of the schedule's purpose, and invisible unless you tabulate it.
    T, e, start = float(cycle_epochs), float(epoch), 0.0
    while e >= T:
        e -= T
        start += T
        nxt = T * cycle_mult
        if horizon > 0:
            remaining = horizon - start
            if remaining <= 0:          # past the horizon: hold the endpoint
                return 0.0
            nxt = min(nxt, remaining)
        T = nxt
    return 0.5 * (1.0 + math.cos(math.pi * (e / T)))


class BaseBrownianModel(L.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.save_hyperparameters(config)
        self.box_size = kwargs.get('box_size', config.get('box_size', 1.0))
        self.n_particles = config.get('n_particles', 7)
        self.n_coords = self.n_particles * 3
        self.latent_dim = config.get('latent_dim', self.n_coords)
        self.noise_rank = config.get('noise_rank', 16)
        self.dt = config.get('dt', 0.0001)
        from nbcore.stochastic_heads import get_stochastic_head
        self.stochastic_head = get_stochastic_head(config)
        self.w_displacement = config.get('w_displacement', 1.0)
        self.w_angle = config.get('w_angle', 1.0)
        self.w_pairwise = config.get('w_pairwise', 1.0)
        # Tail-weighted pairwise-distance term: the equal-weight quantile loss
        # averages the ~2% sub-hard-core tail away, so every model violates the
        # excluded volume. This re-sorts the pooled pairwise distances and
        # upweights the SMALLEST (the violation region) so that tail dominates
        # the gradient -- the CDF-domain form of a low-distance-weighted
        # histogram, without the finite-support escape a soft histogram has.
        # 0.0 = disabled (default; zero behavior change). See compute_pairwise_tail_loss.
        self.w_pairwise_tail = config.get('w_pairwise_tail', 0.0)
        self.pairwise_tail_weight = float(config.get('pairwise_tail_weight', 20.0))
        self.pairwise_tail_power = float(config.get('pairwise_tail_power', 4.0))
        # Excluded-volume hinge: a ONE-SIDED penalty on predicted pairwise distances
        # that fall below the GT hard core = the GT's own minimum pairwise distance,
        # learned as a running min over the target stream (self.ev_core) -> the same
        # term generalizes to any system with NO hardcoded distance. Unlike the
        # tail/quantile match (job 220023 showed it can't remove the ~2% violations,
        # since a distribution-match tolerates a leaking tail), a one-sided hinge
        # enforces the hard support boundary. 0.0 = disabled (default).
        self.w_excluded_volume = config.get('w_excluded_volume', 0.0)
        # Radial-squash supervision, for training the head JOINTLY with the backbone
        # instead of the frozen two-stage route (scripts/membrane/train/train_squash_head.py ->
        # merge_squash_into_ckpt.py). The head is applied inside the AR step either
        # way; this weight is what gives it a gradient. Without it the only signal
        # would be the main losses, and no soft loss has ever built a core -- four
        # unrelated force-free architectures all sit at g(r) onset 7.2.
        self.w_squash = float(config.get('w_squash', 0.0))
        if self.w_squash > 0 and not config.get('squash_enabled', False):
            raise ValueError(
                'w_squash>0 with squash_enabled=False is inert: there is no squash '
                'head to supervise, so the weight would silently do nothing.')
        self.excluded_volume_power = float(config.get('excluded_volume_power', 2.0))
        self.excluded_volume_margin = float(config.get('excluded_volume_margin', 1.0))
        _ev_core = config.get('excluded_volume_core', None)
        self.ev_core_fixed = _ev_core is not None
        self.register_buffer('ev_core', torch.tensor(
            float(_ev_core) if self.ev_core_fixed else float('inf')))
        # The squash target NEEDS a finite core, and ev_core is a running min that is
        # only updated inside compute_excluded_volume_loss -- i.e. only when
        # w_excluded_volume>0. A force-free joint run (the whole point of the squash)
        # would therefore leave it at +inf forever, and minimal_contraction_target with
        # core=inf returns 0.0 EVERYWHERE: the head would be supervised to suppress all
        # inward motion and the model would have no lateral diffusion at all. Demand the
        # core explicitly instead. It is a known constant of the system (GT's own min
        # pairwise distance, 0.0621 fractional here), and a running min would make the
        # supervision depend on which windows happened to be sampled.
        if self.w_squash > 0 and not self.ev_core_fixed:
            raise ValueError(
                'w_squash>0 requires an explicit excluded_volume_core. Without it '
                'ev_core stays +inf (it is only updated by the EV loss), the squash '
                'target collapses to 0 everywhere, and training would suppress all '
                'inward motion. Set excluded_volume_core (fractional, e.g. 0.0621).')
        # --- Excluded-volume FORCE (in-drift repulsion prior) ------------------
        # A physics prior baked into the SDE step: a bounded soft-sphere repulsion
        # added to the per-step displacement so pairs below the GT core are pushed
        # apart at EVERY train+inference step -- unlike the hinge LOSS (a bounded-
        # horizon gradient that never sees the rare collisions of a 40k rollout).
        # Grounded in ML coarse-grained MD, where U = U_prior + U_nn with a
        # repulsion prior on nonbonded distances (CGnet, Wang et al 2019). Bounded
        # linear-spring form, not (sigma/r)^c, for the discrete-jump setting.
        self.ev_force_strength = float(config.get('ev_force_strength', 0.0))
        self.ev_force_enabled = self.ev_force_strength > 0.0
        self.ev_force_power = float(config.get('ev_force_power', 1.0))
        self.ev_force_cutoff = float(config.get('ev_force_cutoff', 1.0))
        self.ev_force_cap = float(config.get('ev_force_cap', 1.5))
        self.ev_force_anneal_epochs = int(config.get('ev_force_anneal_epochs', 0))
        # Schedule shape for the force strength. 'linear' is what the original arm C
        # did and stays the DEFAULT whenever anneal_epochs>0, so every existing config
        # reproduces bit-for-bit without naming the key.
        #   constant        strength held (anneal_epochs is ignored)
        #   linear          strength * (1 - e/E), the original
        #   cosine          strength * 0.5*(1+cos(pi*e/E)), same endpoints, smooth
        #   cosine_restarts SGDR shape: cosine down over a cycle, snap back to full,
        #                   next cycle ev_force_cycle_mult times longer. The point is
        #                   that the model repeatedly loses and regains the crutch
        #                   instead of drifting off it once.
        # ev_force_min_frac floors every schedule, so 'anneal to a residual 10%'
        # is expressible without a separate flag.
        self.ev_force_schedule = str(config.get(
            'ev_force_schedule', 'linear' if self.ev_force_anneal_epochs > 0 else 'constant'))
        _SCHED = ('constant', 'linear', 'cosine', 'cosine_restarts')
        if self.ev_force_schedule not in _SCHED:
            raise ValueError(f"unknown ev_force_schedule={self.ev_force_schedule!r}; "
                             f"expected one of {_SCHED}")
        self.ev_force_cycle_epochs = int(config.get('ev_force_cycle_epochs', 10))
        self.ev_force_cycle_mult = float(config.get('ev_force_cycle_mult', 2.0))
        self.ev_force_min_frac = float(config.get('ev_force_min_frac', 0.0))
        if self.ev_force_schedule == 'cosine_restarts' and self.ev_force_cycle_epochs <= 0:
            raise ValueError('ev_force_schedule=cosine_restarts needs '
                             'ev_force_cycle_epochs > 0')
        self.register_buffer('ev_force_cur', torch.tensor(float(self.ev_force_strength)))
        # --- EV loss: WHICH positions it scores -------------------------------
        # The hinge above is evaluated on the positions the AR loop keeps, and those
        # are POST-force (model_multilag._forward_step_multilag adds ev_force, then
        # wraps). With the force on that is exposure bias: the force has already
        # removed the overlap, so relu(1-d/core) is ~0 on essentially every pair and
        # the penalty gets no gradient from the configurations it exists to teach.
        # Measured on a constructed 7-particle overlap at strength 1.0: ONE force
        # application drops the hinge 84x (3.43e-2 -> 4.06e-4). That is why
        # evc_linear_hinge (w=50000, force annealed to 0) could not work -- the loss
        # was inert exactly while the force was strong, and by the time the force was
        # gone the model had never been taught the core.
        # ev_loss_preforce=True scores the PRE-force positions instead -- supervise the
        # student's raw output, not the teacher's correction -- while the trajectory
        # still continues from the corrected position. Default False so every existing
        # config and checkpoint is bit-for-bit unchanged.
        self.ev_loss_preforce = bool(config.get('ev_loss_preforce', False))
        if self.ev_loss_preforce and self.w_excluded_volume <= 0:
            raise ValueError('ev_loss_preforce=True with w_excluded_volume=0 is inert '
                             '(no EV penalty exists to move onto the pre-force positions)')
        # Capture gate, read by BOTH model_multilag's step (which decides what to
        # return) and its forward (which decides what to stack). One attribute so the
        # two sites cannot disagree about the returned tuple's arity.
        self.ev_capture_preforce = bool(self.ev_loss_preforce and self.w_excluded_volume > 0)
        # --- EV loss: which UNITS ---------------------------------------------
        # 'fractional_mean' (default) is the original  mean_{i<j} relu(1 - d/core)^p
        # with d in box units. 'potential_sum' is  1/2 * sum_{i<j} relu(rc - d)^p  in
        # POSITION units, which at p=2 and ev_force_power=1 is the EXACT potential
        # with -grad U = ev_force. The two differ by the analytic constant
        # 1/2 * core_pos^p * n_pairs, so for core=0.0621, box=856.196, 7 particles
        # (core_pos=53.17, n_pairs=21) w_excluded_volume=1 under 'potential_sum'
        # equals 29684 under 'fractional_mean' -- i.e. the weight then reads as "how
        # many times the force's own potential", and the swept 50000 was 1.68x that.
        self.ev_loss_form = str(config.get('ev_loss_form', 'fractional_mean'))
        _EVFORM = ('fractional_mean', 'potential_sum')
        if self.ev_loss_form not in _EVFORM:
            raise ValueError(f"unknown ev_loss_form={self.ev_loss_form!r}; "
                             f"expected one of {_EVFORM}")
        # --- EV loss WEIGHT schedule: the mirror of ev_force_schedule ----------
        # ev_force_frac ramps the force DOWN; ev_w_frac ramps this multiplier on
        # w_excluded_volume UP over the same window, so the hard core is handed from
        # the force to the penalty rather than both going soft at once. Same four
        # shapes, same maths (_sched_shape). 'constant' (default) pins the multiplier
        # at 1.0 -> the weighted term is bit-identical for every config that does not
        # set these keys. ev_w_overlap_epochs shifts the ramp EARLIER so there is a
        # deliberate band where the force is already weak (violations occur) AND the
        # penalty is already strong; a pure crossfade with no overlap hands over at a
        # single epoch and reproduces the failure mode. See ev_handoff_band.
        self.ev_w_schedule = str(config.get('ev_w_schedule', 'constant'))
        if self.ev_w_schedule not in _SCHED:
            raise ValueError(f"unknown ev_w_schedule={self.ev_w_schedule!r}; "
                             f"expected one of {_SCHED}")
        self.ev_w_ramp_epochs = int(config.get('ev_w_ramp_epochs',
                                               self.ev_force_anneal_epochs))
        self.ev_w_overlap_epochs = float(config.get('ev_w_overlap_epochs', 0.0))
        self.ev_w_min_frac = float(config.get('ev_w_min_frac', 0.0))
        self.ev_w_cycle_epochs = int(config.get('ev_w_cycle_epochs',
                                                self.ev_force_cycle_epochs))
        self.ev_w_cycle_mult = float(config.get('ev_w_cycle_mult',
                                                self.ev_force_cycle_mult))
        if self.ev_w_schedule != 'constant':
            if self.w_excluded_volume <= 0:
                raise ValueError(f'ev_w_schedule={self.ev_w_schedule!r} with '
                                 'w_excluded_volume=0 is inert (nothing to ramp)')
            if self.ev_w_schedule in ('linear', 'cosine') and self.ev_w_ramp_epochs <= 0:
                raise ValueError(f'ev_w_schedule={self.ev_w_schedule!r} needs '
                                 'ev_w_ramp_epochs > 0 (defaults to '
                                 'ev_force_anneal_epochs, which is 0 here)')
            if self.ev_w_schedule == 'cosine_restarts' and self.ev_w_cycle_epochs <= 0:
                raise ValueError('ev_w_schedule=cosine_restarts needs '
                                 'ev_w_cycle_epochs > 0')
        # Plain float, deliberately NOT a buffer: a new state_dict key would make every
        # pre-existing checkpoint fail a strict load. Recomputed from current_epoch each
        # epoch, so a requeued job picks the schedule back up rather than resuming a
        # stale value (same rationale as ev_force_frac).
        self.ev_w_frac_cur = 1.0
        _evtr = torch.triu_indices(self.n_particles, self.n_particles, 1)
        self.register_buffer('ev_pair_i', _evtr[0].clone())
        self.register_buffer('ev_pair_j', _evtr[1].clone())
        self.w_z_quantile = config.get('w_z_quantile', 1.0)
        # Sort z measured from the circular mean of the target's z (minimum image) rather than
        # z wrapped into [0, box): the membrane lies across the box edge, and wrapped it is two bands.
        self.z_quantile_periodic = config.get('z_quantile_periodic', False)
        # 'target_mean': each lag of the displacement, z-displacement, cohesion, depth-step and
        # depth-cohesion terms is divided by the mean size of its target, so the same fractional
        # error counts the same at every lag (as in diffusion/base_model.py).
        self.loss_lag_norm = config.get('loss_lag_norm', 'none')
        if self.loss_lag_norm not in ('none', 'target_mean'):
            raise ValueError(f"loss_lag_norm must be 'none' or 'target_mean', got {self.loss_lag_norm!r}")
        self.w_vacf = config.get('w_vacf', 0.0)
        self.w_cohesion = config.get('w_cohesion', 0.0)
        # Gaussian KL on ln pair distance and on its change over the cohesion lags (logpair_kl, dlogpair_kl)
        self.w_logpair_kl = config.get('w_logpair_kl', 0.0)
        self.w_dlogpair_kl = config.get('w_dlogpair_kl', 0.0)
        # the pairwise and cohesion quantile losses on ln pair distance (compute_logpair_q, compute_dlogpair_q)
        self.w_logpair_q = config.get('w_logpair_q', 0.0)
        self.w_dlogpair_q = config.get('w_dlogpair_q', 0.0)
        self.w_z_displacement = config.get('w_z_displacement', 0.0)
        # Match the signed lag displacement of the time-unwrapped z path rather than
        # |min-image displacement|: the sorted sizes read the same for a path and its mirror.
        self.z_displacement_signed = config.get('z_displacement_signed', False)
        # Same for the lateral term, per axis: sorted sizes cannot see a constant drift along
        # x or y, and a pooled x,y sample would let +x hide behind -y.
        self.displacement_signed = config.get('displacement_signed', False)
        # Lag-wise mean-VACF MSE loss (data-supported replacement / supplement
        # for the cosine-distribution quantile VACF). 0.0 = disabled (default).
        self.w_vacf_mse = config.get('w_vacf_mse', 0.0)
        # Direct latent-velocity autocorrelation loss; 0.0 = disabled (default).
        # Targets corr(dz_t, dz_{t+L}) directly via kernel against a target
        # curve (default = GT position-VACF). Kernel 'tanh' is anti-target
        # (punish decorr + overshoot, no opinion in the good zone);
        # 'mse_target' is plain target-matching.
        self.w_latent_corr = config.get('w_latent_corr', 0.0)
        self.latent_corr_kernel = config.get('latent_corr_kernel', 'tanh')
        self.latent_corr_lags = config.get('latent_corr_lags', None)
        self.latent_corr_target = config.get('latent_corr_target', None)
        self.latent_corr_scale = config.get('latent_corr_scale', 0.05)
        self.w_alpha = config.get('w_alpha', 0.0)
        self.alpha_lags = config.get('alpha_lags', [1, 5, 10, 25, 50, 100, 200, 300])
        self.w_vacf_spatial = config.get('w_vacf_spatial', 0.0)
        # Non-Gaussian parameter (NGP / displacement kurtosis) matching loss;
        # 0.0 = disabled (default). Directly targets the 4th-moment ratio of XY
        # displacement that the quantile/VACF losses leave under-constrained.
        self.w_kurtosis = config.get('w_kurtosis', 0.0)
        self.kurtosis_lags = config.get('kurtosis_lags', [1, 5, 10, 25, 50, 100, 250])
        # Membrane-hypergraph aux loss (predicted node/hyperedge depths vs GT);
        # 0.0 = disabled. Only active when membrane_hypergraph mode is on.
        self.w_membrane = config.get('w_membrane', 0.0)
        # init_head's start depth vs the window's frame-0 depth, MSE. The mem_* terms below see frames 1.. only, so at
        # w_membrane=0 nothing else trains the start depth -- and z_from_depth sets the slab level from it.
        self.w_mem_init = config.get('w_mem_init', 0.0)
        # --- Distributional membrane depth losses: analogs of the coordinate
        # quantile losses applied to the predicted node/hyperedge depth field.
        # All 0.0 = disabled; only active in membrane_hypergraph mode. ---
        self.w_mem_value = config.get('w_mem_value', 0.0)        # depth-value quantile   (<= z_quantile)
        self.w_mem_step = config.get('w_mem_step', 0.0)          # |Δdepth| quantile      (<= displacement)
        self.w_mem_autocorr = config.get('w_mem_autocorr', 0.0)  # Δdepth autocorrelation (<= VACF)
        self.w_mem_pairwise = config.get('w_mem_pairwise', 0.0)  # |d_i-d_j| quantile     (<= pairwise)
        self.w_mem_cohesion = config.get('w_mem_cohesion', 0.0)  # Δ(pairwise diff)       (<= cohesion)
        # Ties depth to GEOMETRY. Every other mem_* term is a POOLED sorted-quantile
        # match, which is invariant to WHICH protein carries which depth -- a model
        # that shuffled protein labels every frame would satisfy all five. GT couples
        # them hard: corr(a protein's mean min-image pair distance, its own depth) =
        # +0.95 / +0.91 on the two trajectories. This is the only term that can see it.
        self.w_mem_coupling = config.get('w_mem_coupling', 0.0)
        # Protein z pulled to the membrane node depth, per frame (_mem_consistency_loss);
        # needs membrane_consistency_slope = box/node_std as well. 0.0 = disabled.
        self.w_mem_consistency = config.get('w_mem_consistency', 0.0)
        self.mem_consistency_slope = config.get('membrane_consistency_slope', 0.0)
        # A protein's z is its node depth plus the start frame's slab level, not a decoder output: in MD z minus
        # depth is one constant over proteins and frames (sd ~1 against a 20-30 depth spread). Set up in
        # model_multilag, which owns the membrane flags.
        self.z_from_depth = bool(config.get('z_from_depth', False))
        # No protein height: the decoder's z step is dropped and every input height is 0, so nothing the model reads,
        # trains on or is scored on has one. The membrane depth carries the vertical (in MD height = depth + a constant).
        self.predict_z = bool(config.get('predict_z', True))
        if not self.predict_z:
            on = sorted(k for k, v in (('w_z_quantile', self.w_z_quantile), ('w_z_displacement', self.w_z_displacement),
                                       ('w_mem_consistency', self.w_mem_consistency),
                                       ('z_from_depth', self.z_from_depth)) if v)
            if on:
                raise ValueError(f'predict_z=False leaves no protein height for {on}')
        self.mem_lags = config.get('mem_lags', [1, 5, 10, 25, 50, 100])
        # mem_step matches the signed depth change instead of its size (as z_displacement_signed)
        self.mem_step_signed = config.get('mem_step_signed', False)
        # --- Membrane loss UNITS ------------------------------------------------
        # 'value' (default) scores the depths exactly as the dataset stores them:
        # divided by their POOLED VALUE std, so the field has std 1 and ONE FRAME of
        # real motion is 0.065 (node) / 0.106 (hyper) of that unit. 'delta' multiplies
        # BOTH prediction and GT by value_std/delta_std (15.39 node, 9.41 hyper)
        # before the loss, putting the loss where a per-frame step is O(1) -- which is
        # the convention the coordinate branch already uses: the displacement loss
        # sees per-frame displacements, never absolute box positions. Depth is a
        # CONFINED process (log-log MSD slope 0.62 against position's 1.09, lag-1
        # value autocorrelation 0.998), so in value units the whole signal a step can
        # move is 1/15th of the quantity being scored and an architecture audit put
        # the depth-loss gradient into the node depths at 1:139 against the position
        # losses. On the pointwise MSE this is arithmetically w_membrane x237
        # (15.39^2) -- but a weight bump is a number someone picked, while this is a
        # units choice that carries to every scale-dependent membrane term at that
        # term's own power, and to node and hyperedge at their own measured ratios.
        # The module's I/O and the npz are untouched: only the loss changes units.
        # Scales are filled in by the subclass that owns the npz (model_multilag).
        self.mem_loss_units = config.get('membrane_loss_units', 'value')
        if self.mem_loss_units not in ('value', 'delta'):
            raise ValueError(f"membrane_loss_units must be 'value' or 'delta', "
                             f"got {self.mem_loss_units!r}")
        self.mem_loss_scale_node = 1.0
        self.mem_loss_scale_hyper = 1.0
        # --- Arms / orientation: excluded volume via lateral protein SHAPE -----
        # The membrane collision is a purely lateral steric wall (verified: GT
        # laterally-close proteins sit at the SAME depth, corr(lateral,|dz|)>0), so z
        # can't prevent it and center-only models leak sub-core overlaps. Each piezo
        # is a rigid 3-fold trimer: predict a per-protein orientation theta (driftless
        # Gaussian, decoupled from translation), hang a rigid 120deg blade template on
        # it, feed the arm geometry into the step (input) and match the extended-point-
        # set distances (loss). arms_enabled=False -> zero behavior change.
        self.arms_enabled = bool(config.get('arms_enabled', False))
        self.arm_radius = float(config.get('arm_radius', 31.73))            # normalized blade radius
        self.arm_sigma_theta = float(config.get('arm_sigma_theta', 0.0332))  # ~1.9deg/frame init
        self.w_arm_pairwise = float(config.get('w_arm_pairwise', 0.0))       # extended-set distance match
        # Delta-theta (in-plane 3-fold spin) distribution match. Applies to BOTH
        # rotation-head families: the 1-DOF arms difference the head's own accumulated
        # theta, the 3-DOF arms difference blade_theta() of the predicted blade state
        # (the same estimator that produced the dataset's stored theta). Without it a
        # 3-DOF arm has NO spin supervision at all -- tilt/rot_step read only the
        # trimer normal, so rotation about that normal is invisible to them.
        self.w_theta_step = float(config.get('w_theta_step', 0.0))
        # Rotation supervision. GT tilt is BOUNDED (mean 15.50 deg, std 7.13, p99 30.0
        # over 7103 frames) while the 3-DOF head is a driftless Gaussian -- at the
        # measured 2.619 deg/step an unconfined walk reaches ~221 deg over a full
        # trajectory. w_tilt supplies the bilayer's restoring statistics; w_rot_step is
        # the 3D analogue of w_theta_step for the NORMAL's step size. Both default
        # 0 = no behaviour change.
        self.w_rot_step = float(config.get('w_rot_step', 0.0))
        self.w_tilt = float(config.get('w_tilt', 0.0))
        # Blade-less IC (LongHorizonStabilityCallback, eval scripts -- no GT blade
        # to hand): 'sampled' draws from the measured GT tilt distribution via
        # sample_orientation (mean 15.50deg, std 7.13deg); all 7 proteins flat and
        # identically oriented is a measure-zero configuration, the 3D restatement
        # of theta=0 for all 7 this codebase already rejects in 2D (see
        # eval_evforce_clustering.py's --init-theta random). 'flat' recovers the
        # old flat 3-fold template for reproducing runs from before this default
        # changed.
        # Validated against BLADE_ICS, which is also what default_blade dispatches
        # through -- one table, so an accepted-but-unimplemented mode is impossible.
        self.blade_ic = str(config.get('blade_ic', 'sampled'))
        if self.blade_ic not in BLADE_ICS:
            raise ValueError(f"unknown blade_ic={self.blade_ic!r}; "
                             f"expected one of {sorted(BLADE_ICS)}")
        # blade_ic='sampled' draws from the ambient global CPU RNG unless it is given a
        # generator, which put UNSEEDED noise on the monitored objective: the stability
        # callback seeds its coordinate IC (ic_seed) but nothing seeded this, and only
        # blade-consuming arms call default_blade -- so moments/se3 carried extra
        # epoch-to-epoch spread (measured std 0.53 on top of the pre-existing 0.94 head
        # noise, ~30% in quadrature) on stab/stability_objective and
        # stab/reported_objective that their own baselines did not. Fixed seed, so the
        # blade-less IC is the same draw every epoch and every process, exactly as
        # LongHorizonStabilityCallback's ic_pool is.
        self.blade_ic_seed = int(config.get('blade_ic_seed', 1234))
        self.w_arm_excluded_volume = float(config.get('w_arm_excluded_volume', 0.0))  # arm-arm hinge (opt)
        self.arm_ev_margin = float(config.get('arm_ev_margin', 1.0))
        self.arm_feat_dim = (self.n_particles * (self.n_particles - 1) // 2) if self.arms_enabled else 0
        # Geometry arm: 'none' = today's behaviour, 'moments' = arm A (covariance
        # appended per protein), 'se3' = arm B (3D triangle + 3-DOF rotation head).
        # Validated against GEOM_ARMS: an unrecognised value ('SE3', 'se3 ', a typo
        # in a sweep json) used to construct cleanly and train a bit-for-bit
        # BASELINE under the arm's wandb group name.
        self.geom_arm = str(config.get('geom_arm', 'none'))
        if self.geom_arm not in GEOM_ARMS:
            raise ValueError(f"unknown geom_arm={self.geom_arm!r}; "
                             f"expected one of {sorted(GEOM_ARMS)}")
        self.geom_spec = GEOM_ARMS[self.geom_arm]
        # A 3-DOF arm tracks a full 3D blade state (rodrigues-advanced) in place of
        # the scalar theta, so it takes the 3D pair feature and skips the 2D
        # theta/extended-point-set losses. Derived from the table, not the string.
        self.geom_rot3d = self.geom_spec['dof'] == 3
        # The per-protein extra features ARE the covariance moments, so the same
        # table entry both widens the encoder and gates the concat -- they can no
        # longer disagree, which is how the concat used to be skipped silently.
        self.geom_moments = self.geom_spec['extra_dim_per_particle'] > 0
        # Both rotation losses score a blade SEQUENCE, which only exists for arms that
        # advance one. On an arm that does not, the weights set cleanly, training_step
        # runs, neither loss key is ever logged, and the run is a bit-for-bit baseline
        # wearing the arm's wandb group name -- the same silent-baseline failure the
        # geom_arm and geom_blade_enabled guards above exist to stop. Checked here
        # rather than beside the weights themselves only because it needs geom_spec.
        if (self.w_rot_step > 0 or self.w_tilt > 0) and not self.geom_spec['advances_blade']:
            raise ValueError(
                f"w_rot_step={self.w_rot_step} / w_tilt={self.w_tilt} require a geom_arm "
                f"that advances a blade state, but geom_arm={self.geom_arm!r} has "
                f"advances_blade=False (see GEOM_ARMS) -- both losses would be silently "
                f"absent. Set w_rot_step=0 and w_tilt=0, or use an advancing arm "
                f"({sorted(k for k, v in GEOM_ARMS.items() if v['advances_blade'])}).")
        self.geom_blade_enabled = bool(config.get('geom_blade_enabled', False))
        self.arm_feat_dim += self.geom_spec['extra_dim_per_particle'] * self.n_particles
        # Arm C widens the encoder input by a GLOBAL embedding, not a per-particle one:
        # the CNN reads one density grid built from all 7 proteins at once, so there is
        # no per-protein row to append. Gated off the table like every other arm.
        self.geom_splat = bool(self.geom_spec.get('needs_splat', False))
        self.splat_out_dim = int(config.get('splat_out_dim', 32)) if self.geom_splat else 0
        self.arm_feat_dim += self.splat_out_dim
        if self.arms_enabled:
            _pt = 4 * self.n_particles                                       # center + 3 arms / protein
            _i, _j = [], []
            for _a in range(_pt):
                for _b in range(_a + 1, _pt):
                    if _a // 4 != _b // 4:                                    # inter-PROTEIN pairs only
                        _i.append(_a); _j.append(_b)
            self.register_buffer('arm28_i', torch.tensor(_i, dtype=torch.long))
            self.register_buffer('arm28_j', torch.tensor(_j, dtype=torch.long))
            _pr, _pc = torch.triu_indices(self.n_particles, self.n_particles, 1)
            self.register_buffer('arm_pair_i', _pr.clone())
            self.register_buffer('arm_pair_j', _pc.clone())
            self.register_buffer('arm_blade', 2 * math.pi * torch.arange(3, dtype=torch.float32) / 3)
        self.use_gradient_checkpointing = config.get('gradient_checkpointing', True)

    def _init_latent_diagnostics(self):
        zero = torch.tensor(0.0, device=self.device)
        return zero, zero, 0

    def _update_latent_diagnostics(self, entropy_sum, top1_sum, count, entropy, top1_prob):
        return entropy_sum + entropy, top1_sum + top1_prob, count + 1

    def _finalize_latent_diagnostics(self, entropy_sum, top1_sum, count):
        if count == 0:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero
        scale = 1.0 / count
        return entropy_sum * scale, top1_sum * scale

    # --- Losses ---
    def compute_quantile_loss_batched(self, pred, target):
        B,T,N = pred.shape
        ps,_ = torch.sort(pred.reshape(B*T,N), dim=0)
        ts,_ = torch.sort(target.reshape(B*T,N), dim=0)
        return (ps-ts).abs().mean()

    def compute_quantile_loss_per_particle(self, pred, target):
        """Sort-match L1 independently per particle, then average across particles."""
        if pred.dim() == 2:
            pred = pred.unsqueeze(-1)
            target = target.unsqueeze(-1)
        if pred.dim() == 3:
            # [B, T, N_particles] -> sort along BxT per particle
            B, T, N = pred.shape
            p_flat = pred.reshape(B * T, N)
            t_flat = target.reshape(B * T, N)
            ps, _ = torch.sort(p_flat, dim=0)
            ts, _ = torch.sort(t_flat, dim=0)
            return (ps - ts).abs().mean()
        return self.compute_quantile_loss_pooled(pred, target)

    def compute_quantile_loss_temporal(self, pred, target, n_windows=4):
        """Sort-match L1 in temporal windows, then average across windows."""
        flat_p = pred.reshape(pred.size(0), pred.size(1), -1) if pred.dim() > 3 else pred
        flat_t = target.reshape(target.size(0), target.size(1), -1) if target.dim() > 3 else target
        if flat_p.dim() == 2:
            flat_p = flat_p.unsqueeze(-1)
            flat_t = flat_t.unsqueeze(-1)
        B, T = flat_p.shape[:2]
        win_size = max(T // n_windows, 1)
        loss = torch.tensor(0.0, device=pred.device)
        count = 0
        for w in range(0, T, win_size):
            pw = flat_p[:, w:w + win_size].reshape(-1)
            tw = flat_t[:, w:w + win_size].reshape(-1)
            if pw.numel() == 0:
                continue
            ps, _ = torch.sort(pw)
            ts, _ = torch.sort(tw)
            if ps.shape[0] != ts.shape[0]:
                ps = F.interpolate(ps.view(1, 1, -1), size=ts.shape[0],
                                   mode='linear', align_corners=False).view(-1)
            loss = loss + (ps - ts).abs().mean()
            count += 1
        return loss / max(count, 1)

    def compute_pairwise_tail_loss(self, pred_d, target_d):
        """Tail-weighted quantile match of pooled pairwise-XY distances. Both are
        flattened and sorted ascending, then |pred_q - gt_q| is weighted so the
        SMALLEST distances (quantile index -> 0) carry weight ~pairwise_tail_weight,
        decaying to 1 at the top with exponent pairwise_tail_power. Because GT's
        smallest pairwise distances sit at the hard core, pushing the predicted
        smallest quantiles up to meet them removes excluded-volume violations;
        the upweighting keeps that tail from being averaged away by the bulk.
        pred_d/target_d: pairwise distances (normalized units), matching shapes."""
        ps, _ = torch.sort(pred_d.reshape(-1))
        ts, _ = torch.sort(target_d.reshape(-1))
        if ps.shape[0] != ts.shape[0]:
            ps = F.interpolate(ps.view(1, 1, -1), size=ts.shape[0],
                               mode='linear', align_corners=False).view(-1)
        i = torch.linspace(0.0, 1.0, ps.shape[0], device=ps.device)   # 0 = smallest distance
        w = 1.0 + (self.pairwise_tail_weight - 1.0) * (1.0 - i) ** self.pairwise_tail_power
        w = w / w.mean()                                              # keep scale comparable to w_pairwise
        return (w * (ps - ts).abs()).mean()

    def compute_excluded_volume_loss(self, pred_d, target_d):
        """One-sided excluded-volume hinge. Penalizes predicted pairwise distances
        below the GT hard core = the GT's own MINIMUM pairwise distance, tracked as a
        running min over the target stream (training only) unless excluded_volume_core
        is set. Normalized by the core -> scale-free, so the same term/weight transfers
        across systems. GT never goes below its own min, so only pred is penalized
        (one-sided) -- this enforces the support boundary a distributional match cannot.
        pred_d/target_d: pairwise distances in fractional (/box_size) units.

        ev_loss_form picks the units. 'fractional_mean' (default) is the original
        mean over pairs of relu(1-d/core)^p. 'potential_sum' is
        1/2 * sum_{i<j} relu(core_pos - d_pos)^p in POSITION units -- at p=2 and
        ev_force_power=1 that is the exact potential whose -grad is ev_force, so the
        weight is expressed in units of "the force's own potential". The two forms
        differ by exactly 1/2 * core_pos^p * n_pairs (pinned in
        tests/membrane/losses/test_ev_teacher_forcing.py)."""
        if not self.ev_core_fixed and self.training:
            with torch.no_grad():
                self.ev_core = torch.minimum(self.ev_core, target_d.detach().min())
        core = self.ev_core * self.excluded_volume_margin
        if not torch.isfinite(core):
            return pred_d.new_zeros(())
        core = core.clamp_min(1e-8)
        pen = torch.relu(1.0 - pred_d / core)                         # fractional penetration
        if self.ev_loss_form == 'potential_sum':
            # relu(core_pos - d_pos) == core_pos * pen. Summed over pairs, not meaned:
            # a potential is extensive in the pairs, and that is what makes the weight
            # comparable to ev_force_strength.
            pen = pen * (core * self.box_size)
            return 0.5 * (pen ** self.excluded_volume_power).sum(dim=-1).mean()
        return (pen ** self.excluded_volume_power).mean()

    # ----- Arms / orientation: reconstruction, input feature, losses -----
    def reconstruct_arms(self, centers_xy, theta):
        """centers_xy [...,N,2], theta [...,N] -> blade xy [...,N,3,2] via the rigid
        symmetric 3-fold template (blades at theta + 120*k, radius arm_radius)."""
        ang = theta.unsqueeze(-1) + self.arm_blade                          # [...,N,3]
        off = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1) * self.arm_radius
        return centers_xy.unsqueeze(-2) + off                               # [...,N,3,2]

    def default_blade(self, batch_size, device=None, dtype=None):
        """[B,N,3,3] blade-less IC -- blade-consuming arms fall back to this when
        the caller has no GT blades (LongHorizonStabilityCallback, the eval
        scripts). Default (blade_ic='sampled') draws per-protein orientations from
        the measured GT tilt distribution via sample_orientation. blade_ic='flat'
        instead returns the old flat 3-fold template (the 3D lift of theta=0, same
        default reconstruct_arms already applies when theta is not supplied) for
        reproducing runs from before 'sampled' became the default.

        Drawn from a LOCAL generator seeded with blade_ic_seed, never the ambient
        global RNG: this IC lands on stab/stability_objective (ModelCheckpoint,
        NaNDetection) and stab/reported_objective (ReduceLROnPlateau, Optuna), and
        only blade-consuming arms reach it, so ambient randomness here is noise that
        the arms under test carry and their baselines do not."""
        dev = self.device if device is None else device
        dt = torch.float32 if dtype is None else dtype
        g = torch.Generator().manual_seed(self.blade_ic_seed)
        return BLADE_ICS[self.blade_ic](batch_size, self.n_particles, self.arm_radius,
                                        generator=g).to(device=dev, dtype=dt)

    def arm_points(self, centers_flat, theta):
        """centers_flat [...,N*3], theta [...,N] -> point cloud [...,4N,2] (center then
        3 blades per protein), ordered so point p belongs to protein p//4."""
        lead = theta.shape[:-1]
        cxy = centers_flat.reshape(*lead, self.n_particles, 3)[..., :2]     # [...,N,2]
        arms = self.reconstruct_arms(cxy, theta)                           # [...,N,3,2]
        pts = torch.cat([cxy.unsqueeze(-2), arms], dim=-2)                  # [...,N,4,2]
        return pts.reshape(*lead, self.n_particles * 4, 2)

    def arm_pair_feature(self, centers_flat, theta):
        """Arm-aware closest-approach: min inter-body XY distance per protein pair,
        normalized. [B,N*3],[B,N] -> [B, n_pairs]. The collision-relevant input signal."""
        B = centers_flat.shape[0]
        four = self.arm_points(centers_flat, theta).reshape(B, self.n_particles, 4, 2)
        pi = four[:, self.arm_pair_i]                                       # [B,P,4,2]
        pj = four[:, self.arm_pair_j]                                       # [B,P,4,2]
        d = pi.unsqueeze(3) - pj.unsqueeze(2)                               # [B,P,4,4,2]
        d = d - self.box_size * torch.round(d / self.box_size)
        dist = d.norm(dim=-1).reshape(B, self.arm_pair_i.shape[0], 16)
        return dist.min(dim=-1).values / self.box_size                     # [B,P]

    def _arm_interbody_dists(self, pts):
        """pts [B,A,4N,2] -> inter-protein point-pair XY min-image distances, normalized."""
        pi = pts[:, :, self.arm28_i]
        pj = pts[:, :, self.arm28_j]
        d = pi - pj
        d = d - self.box_size * torch.round(d / self.box_size)
        return d.norm(dim=-1) / self.box_size                              # [B,A,n_inter]

    def compute_arm_pairwise_loss(self, pred_pts, gt_pts):
        """Distributional match of the extended-point-set inter-body distances (the
        denser near-contact signal the center-only pairwise loss lacks)."""
        qfn = self._get_quantile_fn()
        return qfn(self._arm_interbody_dists(pred_pts), self._arm_interbody_dists(gt_pts))

    def compute_theta_step_loss(self, pred_theta, gt_theta):
        """Match the Delta-theta distribution (driftless Gaussian). pred/gt [B,T,N];
        differenced along time, 3-fold-wrapped (steps are small; guards template flips)."""
        def dstep(th):
            d = th[:, 1:] - th[:, :-1]
            return torch.remainder(d + math.pi / 3, 2 * math.pi / 3) - math.pi / 3
        return self._get_quantile_fn()(dstep(pred_theta), dstep(gt_theta))

    def compute_blade_theta_step_loss(self, pred_blades, gt_blades):
        """In-plane spin supervision for the 3-DOF arms. pred/gt [B,T,P,3,3].

        The 3-DOF head has no scalar theta to difference, so theta is recovered from
        the blade state with blade_theta() -- the SAME 3-fold order parameter
        scripts/membrane/data_prep/build_membrane_arms.py used to write the dataset's `theta` field, so
        this scores the identical observable the 1-DOF arms are scored on, and the two
        families' theta_step_loss numbers are directly comparable.

        Not optional: tilt_angle and rotation_step_angle both read only the trimer
        NORMAL, so rotation about that normal is invisible to them (measured: a 45deg
        per-protein spin about its own normal moves blade coordinates by up to 24.1
        normalized units, 76% of arm_radius, while rotation_step_angle reads 3.3e-6 deg
        and tilt_angle moves 3.8e-6 deg). "No preferred angle" is not "no preferred step
        size": GT spin has a characteristic per-step magnitude, and an unsupervised head
        can drift to its own 0.3 rad = 17.2 deg/step mu bound against GT's ~2.6 deg.

        GT theta is taken from gt_blades rather than the batch's theta_gt so both sides
        go through one estimator; they agree by construction (blade_vec and theta ride
        the same augmentation group element) but a single path cannot drift."""
        return self.compute_theta_step_loss(blade_theta(pred_blades), blade_theta(gt_blades))

    def rotation_losses_enabled(self):
        """True iff any loss that scores a predicted blade SEQUENCE is switched on.

        Two sites must agree exactly: model_multilag.forward's `capture_blades` (which
        decides whether to build and retain the [B,T,P,3,3] predicted sequence) and
        training_step's blade_gt threading (which decides whether to hand forward() the
        GT sequence to score it against). If they disagree in one direction a run pays
        for a tensor nothing reads; in the other a configured weight silently does
        nothing. One helper so they cannot drift.

        w_theta_step only counts for the 3-DOF arms -- the 1-DOF arms get their theta
        from the rotation head's own accumulator and need no blade sequence."""
        return bool(self.w_rot_step > 0 or self.w_tilt > 0
                    or (self.w_theta_step > 0 and self.geom_rot3d))

    def compute_rot_step_loss(self, pred_blades, gt_blades):
        """Match the per-step rotation-magnitude distribution. pred/gt [B,T,P,3,3].

        Sorted-quantile match == 1-Wasserstein for equal sample sizes, which is the
        right form: rotation is stochastic, so only the distribution is predictable.
        geometry_features.rotation_step_angle is gradient-safe at its own degenerate
        point (identical consecutive blades) via its atan2 formulation, so this calls
        it directly rather than keeping a local copy of the geometry."""
        return self._get_quantile_fn()(
            rotation_step_angle(pred_blades[:, :-1], pred_blades[:, 1:]),
            rotation_step_angle(gt_blades[:, :-1], gt_blades[:, 1:]))

    def compute_tilt_loss(self, pred_blades, gt_blades):
        """Match the STATIONARY tilt distribution -- the term that keeps proteins in
        the bilayer. Matching step size alone does not: a driftless walk with the
        correct step size still tumbles. geometry_features.tilt_angle is
        gradient-safe at its own degenerate point (a flat trimer) via its atan2
        formulation, so this calls it directly rather than keeping a local copy."""
        return self._get_quantile_fn()(tilt_angle(pred_blades), tilt_angle(gt_blades))

    def compute_arm_excluded_volume_loss(self, pred_pts, gt_pts):
        """Optional one-sided hinge on inter-body distances below the GT arm-arm min."""
        pd = self._arm_interbody_dists(pred_pts)
        core = (self._arm_interbody_dists(gt_pts).min().detach() * self.arm_ev_margin).clamp_min(1e-8)
        return (torch.relu(1.0 - pd / core) ** 2).mean()

    def ev_force(self, x_flat):
        """Bounded soft-sphere excluded-volume repulsion, in POSITION units, added
        to the per-step displacement (the U_prior half of U=U_prior+U_nn). For each
        xy min-image pair below the core, push the two particles apart by
        strength * overlap (linear spring, capped). z is untouched. Branch-free and
        buffer-driven so it stays inside the torch.compile'd step; enable/disable is
        gated at the call site by self.ev_force_enabled (a trace-time constant).
        x_flat: [B, N*3] positions (any wrap; min-image is wrap-invariant)."""
        core = (self.ev_core * self.box_size).clamp_min(1e-6)         # fractional -> position units
        N = self.n_particles
        B = x_flat.shape[0]
        xy = x_flat.view(B, N, 3)[..., :2]                           # [B,N,2]
        diff = xy.unsqueeze(2) - xy.unsqueeze(1)                     # [B,N,N,2] i<-j
        diff = diff - self.box_size * torch.round(diff / self.box_size)
        d = diff.norm(dim=-1, keepdim=True).clamp_min(1e-6)          # [B,N,N,1]
        rc = core * self.ev_force_cutoff
        overlap = torch.relu(rc - d)                                 # [B,N,N,1] position units
        eye = torch.eye(N, device=xy.device, dtype=torch.bool).view(1, N, N, 1)
        overlap = overlap.masked_fill(eye, 0.0)                      # drop self-pairs
        mag = self.ev_force_cur * core * (overlap / core).pow(self.ev_force_power)
        kick = ((diff / d) * mag).sum(dim=2)                         # [B,N,2] repulsion on i
        kn = kick.norm(dim=-1, keepdim=True).clamp_min(1e-9)         # cap per-particle kick norm
        kick = kick * (torch.minimum(kn, self.ev_force_cap * core) / kn)
        out = torch.zeros(B, N, 3, device=xy.device, dtype=xy.dtype)
        out[..., :2] = kick
        return out.view(B, N * 3)

    def ev_force_frac(self, epoch):
        """Multiplier on ev_force_strength at `epoch`, in [ev_force_min_frac, 1].

        Pure function of the epoch so it can be tested and plotted without a trainer,
        and so a REQUEUED job recomputes the same value from current_epoch rather than
        resuming whatever was last written into the buffer.
        """
        lo = self.ev_force_min_frac
        f = _sched_shape(self.ev_force_schedule, epoch, self.ev_force_anneal_epochs,
                         self.ev_force_cycle_epochs, self.ev_force_cycle_mult)
        return lo + (1.0 - lo) * f

    def ev_w_frac(self, epoch):
        """Multiplier on w_excluded_volume at `epoch`, in [ev_w_min_frac, 1] and
        NON-DECREASING -- the mirror of ev_force_frac.

        Same shape function, evaluated ev_w_overlap_epochs EARLIER and then flipped.
        So the penalty reaches full strength at epoch
        (ev_w_ramp_epochs - ev_w_overlap_epochs), while the force is still on. That
        band is the entire point: a pure crossfade (overlap 0, shared horizon) makes
        the penalty strong only once the force is already gone, which is the failure
        mode -- the loss never sees a violation it could still correct cheaply.

        Pure function of the epoch, and reads nothing off self beyond its own config,
        for the same two reasons ev_force_frac is (testable/plottable without a
        trainer; a requeued job recomputes rather than resumes).
        """
        if self.ev_w_schedule == 'constant':
            return 1.0
        lo = self.ev_w_min_frac
        f = _sched_shape(self.ev_w_schedule, epoch + self.ev_w_overlap_epochs,
                         self.ev_w_ramp_epochs, self.ev_w_cycle_epochs,
                         self.ev_w_cycle_mult)
        return lo + (1.0 - lo) * (1.0 - f)

    def ev_handoff_band(self, max_epochs, force_thresh=0.5, weight_thresh=0.5):
        """Epochs where the force is already WEAKENED (0 < frac <= force_thresh) and
        the penalty is already STRONG (>= weight_thresh) -- i.e. where violations
        actually occur and something is punishing them.

        Non-empty means the two schedules overlap; empty means a gapped crossfade,
        which is exactly what evc_linear_hinge did. Used by the tests and by the
        evtf launcher's preflight so a config with an empty band cannot be submitted.
        """
        return [e for e in range(int(max_epochs))
                if 0.0 < self.ev_force_frac(e) <= force_thresh
                and self.ev_w_frac(e) >= weight_thresh]

    def on_train_epoch_start(self):
        # Force schedule (arm C family): the force guarantees exclusion early, then
        # fades so the network must stand on its own. Shape comes from
        # ev_force_schedule; 'constant' leaves it at full strength (arm A).
        if self.ev_force_enabled:
            self.ev_force_cur.fill_(self.ev_force_strength
                                    * self.ev_force_frac(self.current_epoch))
        # Mirror ramp on the penalty weight (ev_w_frac). 'constant' returns 1.0, so
        # configs that never set ev_w_schedule keep the flat weight they always had.
        if self.w_excluded_volume > 0:
            self.ev_w_frac_cur = self.ev_w_frac(self.current_epoch)

    def _get_quantile_fn(self):
        """Return the quantile loss function based on config."""
        mode = self.hparams.get('quantile_mode', 'pooled')
        if mode == 'per_particle':
            return self.compute_quantile_loss_per_particle
        elif mode == 'temporal':
            return self.compute_quantile_loss_temporal
        elif mode == 'both':
            def _both(pred, target):
                return (self.compute_quantile_loss_per_particle(pred, target) +
                        self.compute_quantile_loss_temporal(pred, target)) * 0.5
            return _both
        return self.compute_quantile_loss_pooled

    def compute_quantile_loss_masked(self, pred, target, active_mask):
        """Per-sample quantile loss with particle masking. Used in multi_system mode."""
        B = pred.size(0)
        loss = torch.tensor(0.0, device=pred.device)
        for i in range(B):
            mask = active_mask[i]  # [P]
            if pred[i].dim() == 2:
                p_i = pred[i][:, mask].reshape(-1)
                t_i = target[i][:, mask].reshape(-1)
            else:
                p_i = pred[i][mask].reshape(-1)
                t_i = target[i][mask].reshape(-1)
            if p_i.numel() == 0:
                continue
            ps, _ = torch.sort(p_i)
            ts, _ = torch.sort(t_i)
            if ps.shape[0] != ts.shape[0]:
                ps = F.interpolate(ps.view(1, 1, -1), size=ts.shape[0],
                                   mode='linear', align_corners=False).view(-1)
            loss = loss + (ps - ts).abs().mean()
        return loss / max(B, 1)

    def compute_quantile_loss_pooled(self, pred, target):
        ps, _ = torch.sort(pred.reshape(-1))
        ts, _ = torch.sort(target.reshape(-1))
        if ps.shape[0] != ts.shape[0]:
            ps = F.interpolate(
                ps.view(1, 1, -1), size=ts.shape[0],
                mode='linear', align_corners=False,
            ).view(-1)
        return (ps - ts).abs().mean()

    def compute_weighted_quantile_loss_pooled(self, pred, target, tail='low',
                                              beta=2.0, d_ref=None, eps=1.0,
                                              max_weight=1e3):
        # Sort-match L1 with per-rank weighting driven by the (detached) sorted target.
        # tail='low'  => w = (d_ref / (ts + eps))^beta  -- amplifies errors at small target values
        # tail='high' => w = ((ts + eps) / d_ref)^beta  -- amplifies errors at large target values
        # d_ref defaults to the sorted-target median so weights are scale-aware per call.
        ps, _ = torch.sort(pred.reshape(-1))
        ts, _ = torch.sort(target.reshape(-1))
        if ps.shape[0] != ts.shape[0]:
            ps = F.interpolate(
                ps.view(1, 1, -1), size=ts.shape[0],
                mode='linear', align_corners=False,
            ).view(-1)
        if d_ref is None:
            d_ref_t = ts[ts.shape[0] // 2].detach().clamp_min(eps)
        else:
            d_ref_t = torch.as_tensor(d_ref, device=ts.device, dtype=ts.dtype)
        if tail == 'low':
            w = (d_ref_t / (ts + eps)).clamp(max=max_weight) ** beta
        elif tail == 'high':
            w = ((ts + eps) / d_ref_t).clamp(max=max_weight) ** beta
        else:
            raise ValueError(f"Unknown tail: {tail!r}; expected 'low' or 'high'")
        w = w.detach()
        return ((ps - ts).abs() * w).mean()

    def _particle_mask_to_pair_mask(self, active_mask):
        """Convert particle-level mask [B, P] to pair-level mask [B, n_pairs].

        A pair (i, j) is active iff both particles i and j are active.
        Uses the same triu_rows/triu_cols as geo_embed.
        """
        rows = self.geo_embed.triu_rows  # [n_pairs]
        cols = self.geo_embed.triu_cols   # [n_pairs]
        # active_mask: [B, P] -> index particles at pair endpoints
        return active_mask[:, rows] & active_mask[:, cols]  # [B, n_pairs]

    def _drop_z(self, x):
        """x [..., >= n_coords] with every protein's z at 0 when predict_z is off; multiplied, so a dropped z step
        passes no gradient back to whatever made it"""
        if self.predict_z:
            return x
        keep = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        keep[2:self.n_coords:3] = 0
        return x * keep

    def _unwrap_xy(self, positions, active_mask=None):
        xy = positions[:,:,:self.n_coords].reshape(positions.size(0),-1,self.n_particles,3)[:,:,:,:2]
        d = xy[:,1:]-xy[:,:-1]; d = d - self.box_size*torch.round(d/self.box_size)
        uw = torch.zeros_like(xy); uw[:,0]=xy[:,0]; uw[:,1:]=xy[:,0:1]+torch.cumsum(d,dim=1)
        if active_mask is not None:
            mask_2d = active_mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, P, 1]
            uw = uw * mask_2d
        return uw

    def _unwrap_t(self, x):
        d = x[:,1:]-x[:,:-1]; d = d - self.box_size*torch.round(d/self.box_size)
        return torch.cat([x[:,:1], x[:,:1]+torch.cumsum(d,dim=1)], dim=1)

    def _lag_scale(self, target, active_mask=None):
        """loss_lag_norm: mean size of one lag's target. None when the norm is off."""
        if self.loss_lag_norm == 'none':
            return None
        if active_mask is not None:
            raise NotImplementedError("loss_lag_norm is not implemented for masked (multi-system) batches")
        return target.detach().abs().mean().clamp_min(1e-8)

    def _lag_add(self, loss, scale, value, target, active_mask=None):
        s = self._lag_scale(target, active_mask)
        return (loss + value, scale) if s is None else (loss + value/s, scale + s)

    def _lag_mean(self, loss, scale, count):
        if count and self.loss_lag_norm != 'none':
            return loss/count*(scale/count)
        return loss/max(count, 1)

    def compute_z_quantile_loss(self, pz, tz, active_mask=None):
        """pz, tz [B,T,P]: z as a box fraction in [0, 1), sort-matched."""
        if self.z_quantile_periodic:
            a = tz.detach() * (2*math.pi)
            if active_mask is not None:
                a = a[active_mask.unsqueeze(1).expand_as(a)]
            zc = torch.atan2(a.sin().mean(), a.cos().mean()) / (2*math.pi)
            pz, tz = pz - zc, tz - zc
            pz, tz = pz - torch.round(pz), tz - torch.round(tz)
        if active_mask is not None:
            return self.compute_quantile_loss_masked(pz, tz, active_mask)
        return self._get_quantile_fn()(pz, tz)

    # --- Membrane depth-field distributional losses ------------------------------
    # Operate on a depth sequence [B,T,D] (D = 7 nodes or 120 hyperedges), treating
    # the D depths as the "coordinate" columns and reusing the sort-match quantile
    # machinery. Analogs of the coordinate losses named in each docstring.
    def _mem_step_qloss(self, pred, target):
        """|Δdepth| over lags, quantile-matched (analog of displacement)."""
        loss = torch.tensor(0.0, device=pred.device); n = 0; scale = 0.0
        T = pred.size(1)
        for L in self.mem_lags:
            if L >= T:
                break
            dp = pred[:, L:] - pred[:, :-L]
            dt = target[:, L:] - target[:, :-L]
            if not self.mem_step_signed:
                dp, dt = dp.abs(), dt.abs()
            loss, scale = self._lag_add(loss, scale, self.compute_quantile_loss_pooled(dp, dt), dt); n += 1
        return self._lag_mean(loss, scale, n)

    def _mem_autocorr_loss(self, pred, target):
        """Scalar temporal autocorrelation of Δdepth over lags, matched pred<->GT
        (analog of VACF; a scalar depth has no direction, so this is a Pearson-style
        autocorrelation curve, not the cosine-vector VACF)."""
        def _acurve(x):
            v = x[:, 1:] - x[:, :-1]                                  # [B,T-1,D]
            v = v - v.mean(dim=1, keepdim=True)
            var = (v * v).mean(dim=1).clamp_min(1e-8)                 # [B,D]
            Tm = v.size(1); cs = []
            for L in self.mem_lags:
                if L >= Tm:
                    break
                c = (v[:, L:] * v[:, :-L]).mean(dim=1) / var          # [B,D]
                cs.append(c.mean())
            return torch.stack(cs) if cs else torch.zeros(1, device=x.device)
        return ((_acurve(pred) - _acurve(target)) ** 2).mean()

    def _mean_pair_distance(self, pos):
        """Per-protein mean min-image XY distance to the other P-1 proteins. [B,T,P]."""
        B, T = pos.shape[0], pos.shape[1]
        xy = pos[..., :self.n_coords].reshape(B, T, self.n_particles, 3)[..., :2]
        d = xy[:, :, :, None, :] - xy[:, :, None, :, :]
        d = d - self.box_size * torch.round(d / self.box_size)
        dist = torch.sqrt((d ** 2).sum(-1) + 1e-12)
        off = 1.0 - torch.eye(self.n_particles, device=pos.device)
        return (dist * off).sum(-1) / (self.n_particles - 1)

    @staticmethod
    def _frame_corr(a, b):
        """Correlation of a and b across the PROTEIN axis, per (batch, frame). [B,T]."""
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        num = (a * b).mean(dim=-1)
        den = a.std(dim=-1).clamp_min(1e-6) * b.std(dim=-1).clamp_min(1e-6)
        return num / den

    def _mem_coupling_qloss(self, node_pred, node_gt, pred_pos, target_pos):
        """Tie each protein's depth to its own isolation, within the frame.

        Every other mem_* term is a pooled sorted-quantile match, and sorting
        destroys which protein at which timestep carries which depth -- so all five
        are satisfied exactly by a model that reproduces the depth HISTOGRAM with the
        assignment scrambled. This is the third place that blind spot has shown up
        (after the pooled coordinate quantiles and the depth objective).

        Measured on the GT npz directly, 2026-09-13: pooled corr(mean min-image
        XY pair distance, own depth) = +0.951 traj1 / +0.904 traj2, and -- the part
        that matters here -- the WITHIN-FRAME rank correlation over the 7 proteins is
        +0.87 / +0.83. Isolation predicts depth almost deterministically, and nothing
        in the loss set can currently see it.

        Per-frame correlation rather than a regression: it is scale- and offset-free,
        so it neither fights mem_value (which owns the level) nor cares which of the
        two membrane_loss_units modes is active. It is invariant to a JOINT
        relabelling of the proteins -- required, since the S7 augmentation applies
        one -- but not to scrambling depth against distance, which is the failure.
        Quantile-matching the [B,A] correlations rather than their mean keeps the
        whole profile: GT's coupling is not constant in time and a mean-match would
        let the model sit at the average.
        """
        cp = self._frame_corr(self._mean_pair_distance(pred_pos), node_pred)
        cg = self._frame_corr(self._mean_pair_distance(target_pos), node_gt)
        return self.compute_quantile_loss_pooled(cp, cg.detach())

    def _mem_consistency_loss(self, node, pz):
        """Protein z must agree with the node depth, frame by frame.

        In MD the two are one quantity: a protein's min-image height relative to the
        other six against its node depth relative to theirs has slope 0.98-1.00 and
        r 0.994-0.996 on all three arms npz (2026-10-23). The model learned the depth
        side (mem_coupling ties it to isolation) but its z ignored it, r ~0 against
        isolation, since the z terms are pooled quantiles that cannot see which
        protein sits where. Centring over the proteins removes the slab's level and
        any per-frame offset; the min-image step keeps a slab that straddles the z
        wrap in one piece. The depth is detached so the term moves z to the depth
        module, never the depth to z. node [B,A,P] normalised depth, pz [B,A,P] z/box;
        mem_consistency_slope = box/node_std converts one to the other.
        """
        h = pz - pz[..., :1]
        h = h - torch.round(h)
        hc = h - h.mean(-1, keepdim=True)
        n = node.detach()
        nc = n - n.mean(-1, keepdim=True)
        return ((nc - self.mem_consistency_slope * hc) ** 2).mean()

    def _mem_pairwise_qloss(self, pred, target):
        """Distribution of |d_i - d_j| across the D depths, quantile-matched
        (analog of pairwise distance)."""
        D = pred.size(-1)
        iu = torch.triu_indices(D, D, offset=1, device=pred.device)
        pd = (pred[..., iu[0]] - pred[..., iu[1]]).abs()
        td = (target[..., iu[0]] - target[..., iu[1]]).abs()
        return self.compute_quantile_loss_pooled(pd, td)

    def _mem_cohesion_qloss(self, pred, target):
        """Change over time of the pairwise depth differences, quantile-matched
        (analog of cohesion)."""
        D = pred.size(-1)
        iu = torch.triu_indices(D, D, offset=1, device=pred.device)
        pd = (pred[..., iu[0]] - pred[..., iu[1]]).abs()
        td = (target[..., iu[0]] - target[..., iu[1]]).abs()
        loss = torch.tensor(0.0, device=pred.device); n = 0; scale = 0.0
        T = pd.size(1)
        for L in self.mem_lags:
            if L >= T:
                break
            dt = td[:, L:] - td[:, :-L]
            loss, scale = self._lag_add(
                loss, scale, self.compute_quantile_loss_pooled(pd[:, L:] - pd[:, :-L], dt), dt); n += 1
        return self._lag_mean(loss, scale, n)

    def compute_displacement_quantile_loss(self, pp, tp, lags=None,
                                            pred_xy_unwrapped=None, target_xy_unwrapped=None,
                                            active_mask=None):
        if lags is None: lags=[1,5,10,25,50,100,200,300,500,1000,2000,3000]
        loss=torch.tensor(0.0,device=pp.device); count=0; scale=0.0
        pxy = pred_xy_unwrapped if pred_xy_unwrapped is not None else self._unwrap_xy(pp, active_mask)
        txy = target_xy_unwrapped if target_xy_unwrapped is not None else self._unwrap_xy(tp, active_mask)
        qfn = self._get_quantile_fn()
        match = qfn if active_mask is None else (
            lambda p, t: self.compute_quantile_loss_masked(p, t, active_mask))
        for lag in lags:
            if pxy.size(1)<=lag: continue
            pd, td = pxy[:,lag:]-pxy[:,:-lag], txy[:,lag:]-txy[:,:-lag]
            if self.displacement_signed:
                v = 0.5*(match(pd[...,0], td[...,0]) + match(pd[...,1], td[...,1]))
            else:
                pd, td = torch.sqrt((pd**2).sum(-1)+1e-6), torch.sqrt((td**2).sum(-1)+1e-6)
                v = match(pd, td)
            loss, scale = self._lag_add(loss, scale, v, td, active_mask)
            count+=1
        return self._lag_mean(loss, scale, count)

    def compute_z_displacement_quantile_loss(self, pp, tp, lags=None, active_mask=None):
        if lags is None: lags=[1,5,10,25,50,100,200,300,500,1000,2000,3000]
        loss=torch.tensor(0.0,device=pp.device); count=0; scale=0.0
        B = pp.size(0)
        pz = pp[:,:,:self.n_coords].reshape(B,-1,self.n_particles,3)[:,:,:,2]
        tz = tp[:,:,:self.n_coords].reshape(B,-1,self.n_particles,3)[:,:,:,2]
        qfn = self._get_quantile_fn()
        signed = self.z_displacement_signed
        if signed:
            # unwrap along time first: a min-image difference at lag L changes sign once
            # the path has moved half a box inside the lag
            pz, tz = self._unwrap_t(pz), self._unwrap_t(tz)
        for lag in lags:
            if pz.size(1)<=lag: continue
            pd = pz[:,lag:]-pz[:,:-lag]
            td = tz[:,lag:]-tz[:,:-lag]
            if not signed:
                pd = (pd - self.box_size*torch.round(pd/self.box_size)).abs()
                td = (td - self.box_size*torch.round(td/self.box_size)).abs()
            v = self.compute_quantile_loss_masked(pd, td, active_mask) if active_mask is not None else qfn(pd,td)
            loss, scale = self._lag_add(loss, scale, v, td, active_mask)
            count+=1
        return self._lag_mean(loss, scale, count)

    def _get_xy_turning_angles(self, positions, lag=1, xy_unwrapped=None, active_mask=None):
        xy = xy_unwrapped if xy_unwrapped is not None else self._unwrap_xy(positions)
        v=xy[:,lag:]-xy[:,:-lag]; v1,v2=v[:,:-lag],v[:,lag:]
        if v1.size(1)==0: return None
        dot=(v1*v2).sum(-1); cross=v1[...,0]*v2[...,1]-v1[...,1]*v2[...,0]
        return torch.atan2(cross.abs(), dot)

    def compute_angle_quantile_loss(self, pp, tp, lags=None,
                                     pred_xy_unwrapped=None, target_xy_unwrapped=None,
                                     active_mask=None):
        if lags is None: lags=[1,5,10,25,50,100,200,300,500,1000,2000,3000]
        loss=torch.tensor(0.0,device=pp.device); count=0
        qfn = self._get_quantile_fn()
        for lag in lags:
            if pp.size(1)<=lag: continue
            pa=self._get_xy_turning_angles(pp,lag,pred_xy_unwrapped,active_mask=active_mask)
            ta=self._get_xy_turning_angles(tp,lag,target_xy_unwrapped,active_mask=active_mask)
            if pa is None or ta is None or pa.size(1)<2 or ta.size(1)<2: continue
            if active_mask is not None:
                loss = loss + self.compute_quantile_loss_masked(pa, ta, active_mask)
            else:
                loss=loss+qfn(pa,ta)
            count+=1
        return loss/max(count,1)

    def compute_logpair_q(self, pred_pw, tgt_pw, floor=LOG_FLOOR):
        """the pairwise quantile loss on ln(distance), which sees the whole shape (empty core, contact peak) at
        every scale, and the signed offset of the prediction's mean ln distance (+ proteins further apart)"""
        lp, lt = _ln_dist(pred_pw, floor), _ln_dist(tgt_pw, floor)
        return self._get_quantile_fn()(lp, lt), lp.mean() - lt.mean()

    def compute_dlogpair_q(self, pred_pw, tgt_pw, lags=DLOG_LAGS, floor=LOG_FLOOR):
        """the cohesion loss on ln(distance): sort-matches the signed change in ln(distance) over each lag that fits
        the window, with the same lag normalisation, and the mean signed offset of the changes (+ drifting apart)"""
        lp, lt = _ln_dist(pred_pw, floor), _ln_dist(tgt_pw, floor)
        T = min(lp.size(1), lt.size(1))
        qfn = self._get_quantile_fn()
        loss, scale, count, off = lp.new_zeros(()), 0.0, 0, []
        for L in lags:
            if L >= T:
                continue
            dp, dt = lp[:, L:T] - lp[:, :T - L], lt[:, L:T] - lt[:, :T - L]
            loss, scale = self._lag_add(loss, scale, qfn(dp, dt), dt)
            off.append(dp.mean() - dt.mean())
            count += 1
        return self._lag_mean(loss, scale, count), (torch.stack(off).mean() if off else lp.new_zeros(()))

    def compute_cohesion_loss(self, pred_positions, target_positions, lags=None,
                              pred_pw=None, tgt_pw=None, active_mask=None):
        """
        Penalizes divergence in pairwise distance *changes* over time.

        For each lag, computes the change in each pairwise XY distance:
            Δd_ij(t, lag) = d_ij(t+lag) - d_ij(t)
        Then quantile-matches these Δd distributions between pred and GT.

        This directly penalizes cluster dissolution: if pred particles fly apart,
        their Δd will be systematically positive while GT's Δd fluctuates around zero.

        Uses wrapped coordinates + PBC — no unwrapping needed.
        """
        if lags is None:
            lags = [1,5,10,25,50,100,200,300,500,1000,2000,3000]
        loss = torch.tensor(0.0, device=pred_positions.device)
        count = 0
        scale = 0.0

        # Pairwise distances at each timestep: [B, T, n_pairs]
        if pred_pw is None:
            pred_pw = self.geo_embed.compute_distances_xy_only(pred_positions)
        if tgt_pw is None:
            tgt_pw = self.geo_embed.compute_distances_xy_only(target_positions)

        qfn = self._get_quantile_fn()
        for lag in lags:
            if pred_pw.size(1) <= lag or tgt_pw.size(1) <= lag:
                continue
            # Change in pairwise distance over lag: [B, T-lag, n_pairs]
            pred_delta_d = pred_pw[:, lag:] - pred_pw[:, :-lag]
            tgt_delta_d = tgt_pw[:, lag:] - tgt_pw[:, :-lag]
            if active_mask is not None:
                v = self.compute_quantile_loss_masked(pred_delta_d, tgt_delta_d, active_mask)
            else:
                v = qfn(pred_delta_d, tgt_delta_d)
            loss, scale = self._lag_add(loss, scale, v, tgt_delta_d, active_mask)
            count += 1

        return self._lag_mean(loss, scale, count)

    def compute_vacf_quantile_loss(self, pred_positions, target_positions,
                                   pred_xy_unwrapped=None, target_xy_unwrapped=None):
        """Quantile-match VACF cosine similarity distributions across multiple lags."""
        pred_xy = pred_xy_unwrapped if pred_xy_unwrapped is not None else self._unwrap_xy(pred_positions)
        target_xy = target_xy_unwrapped if target_xy_unwrapped is not None else self._unwrap_xy(target_positions)

        # _unwrap_xy returns [B, T, N, 2] — reshape if it returns [B, T, N*2]
        if pred_xy.dim() == 3:
            pred_xy = pred_xy.reshape(pred_xy.size(0), pred_xy.size(1), self.n_particles, 2)
            target_xy = target_xy.reshape(target_xy.size(0), target_xy.size(1), self.n_particles, 2)

        from membrane.vacf_loss import compute_vacf_quantile_loss as _vacf_ql
        return _vacf_ql(pred_xy, target_xy, self.compute_quantile_loss_batched)

    def compute_vacf_mse_loss(self, pred_positions, target_positions,
                              pred_xy_unwrapped=None, target_xy_unwrapped=None,
                              lags=None, velocity_lag=1):
        """Lag-wise mean-VACF MSE; particle-blind (pooled)."""
        pred_xy = pred_xy_unwrapped if pred_xy_unwrapped is not None else self._unwrap_xy(pred_positions)
        target_xy = target_xy_unwrapped if target_xy_unwrapped is not None else self._unwrap_xy(target_positions)
        if pred_xy.dim() == 3:
            pred_xy = pred_xy.reshape(pred_xy.size(0), pred_xy.size(1), self.n_particles, 2)
            target_xy = target_xy.reshape(target_xy.size(0), target_xy.size(1), self.n_particles, 2)
        from membrane.vacf_loss import compute_vacf_mse_loss as _vacf_mse
        return _vacf_mse(pred_xy, target_xy, lags=lags, velocity_lag=velocity_lag)

    def compute_alpha_loss(self, pred_xy_unwrapped, target_xy_unwrapped, lags=None):
        """MSE between predicted and target anomalous diffusion exponent α.

        α is the slope of log(MSD) vs log(lag), computed via differentiable
        closed-form least squares.  Both pred and target α are computed from
        the current batch so the comparison is self-consistent across window
        lengths.
        """
        if lags is None:
            lags = self.alpha_lags
        T = pred_xy_unwrapped.size(1)
        usable = [l for l in lags if l < T]
        if len(usable) < 2:
            return torch.tensor(0.0, device=pred_xy_unwrapped.device)

        log_lags = torch.tensor([math.log(l) for l in usable],
                                device=pred_xy_unwrapped.device)

        def _alpha(xy):
            # xy: [B, T, N, 2]
            log_msds = []
            for lag in usable:
                disp = xy[:, lag:] - xy[:, :-lag]       # [B, T-lag, N, 2]
                msd = (disp ** 2).sum(-1).mean(dim=(1, 2))  # [B]
                log_msds.append(torch.log(msd.clamp(min=1e-8)))
            log_msd = torch.stack(log_msds, dim=1)     # [B, n_lags]
            n = len(usable)
            sx = log_lags.sum()
            sxx = (log_lags ** 2).sum()
            sy = log_msd.sum(dim=1)                     # [B]
            sxy = (log_msd * log_lags.unsqueeze(0)).sum(dim=1)  # [B]
            return (n * sxy - sx * sy) / (n * sxx - sx ** 2)    # [B]

        alpha_pred = _alpha(pred_xy_unwrapped)
        alpha_tgt = _alpha(target_xy_unwrapped)
        return ((alpha_pred - alpha_tgt) ** 2).mean()

    def compute_kurtosis_loss(self, pred_xy_unwrapped, target_xy_unwrapped, lags=None):
        """MSE between predicted and target non-Gaussian parameter (NGP, 2D XY).

        NGP(L) = <r4>/(2 <r2>^2) - 1 for d=2, matching the eval metric in
        membrane/physics_metrics.py exactly (r2 = squared XY displacement at lag
        L).  Pooled per-lag scalar (the 4th moment needs many samples), MSE over
        lags -- structurally identical to compute_alpha_loss.  Differentiable
        through the prediction; the GT term carries no gradient.
        """
        if lags is None:
            lags = self.kurtosis_lags
        T = pred_xy_unwrapped.size(1)
        usable = [l for l in lags if l < T]
        if len(usable) == 0:
            return torch.tensor(0.0, device=pred_xy_unwrapped.device)
        loss = torch.tensor(0.0, device=pred_xy_unwrapped.device)
        for lag in usable:
            pr2 = ((pred_xy_unwrapped[:, lag:] - pred_xy_unwrapped[:, :-lag]) ** 2).sum(-1)
            tr2 = ((target_xy_unwrapped[:, lag:] - target_xy_unwrapped[:, :-lag]) ** 2).sum(-1)
            ngp_p = pr2.pow(2).mean() / (2.0 * pr2.mean().pow(2) + 1e-12) - 1.0
            ngp_t = tr2.pow(2).mean() / (2.0 * tr2.mean().pow(2) + 1e-12) - 1.0
            loss = loss + (ngp_p - ngp_t) ** 2
        return loss / len(usable)

    def compute_vacf_spatial_l1_loss(self, pred_xy_unwrapped, target_xy_unwrapped):
        from membrane.vacf_loss import compute_vacf_spatial_l1_loss as _vsl1
        return _vsl1(pred_xy_unwrapped, target_xy_unwrapped)

    # --- Forward step ---
    def _forward_step(self, curr_x, enc_state, z_prev, curr_emb, ns1=None, ns2=None, eps_prev=None):
        transition_params, enc_new = self.encoder(curr_emb, enc_state)
        transition_params = squeeze_transition_params(transition_params, dim=1)
        noise_state = (ns1, ns2) if ns1 is not None else None
        z_new, latent_entropy, latent_top1_prob, noise_state = self.stochastic_head.step(
            transition_params, z_prev, noise_state=noise_state)
        ds,dm,dsig,eps_t = self.decoder(z_new, curr_emb, eps_prev=eps_prev)
        xp = torch.remainder(curr_x+ds, self.box_size)
        # eps_t (decoder AR(1) state; None when white) is always returned last.
        if noise_state is not None:
            return xp, ds, dm, dsig, enc_new, z_new, latent_entropy, latent_top1_prob, noise_state[0], noise_state[1], eps_t
        return xp, ds, dm, dsig, enc_new, z_new, latent_entropy, latent_top1_prob, eps_t

    # --- Forward ---
    def forward(self, x_sequence, target_sequence=None,
                active_mask=None, edge_src=None, edge_dst=None, edge_mask=None):
        if self.hparams.get('geo_ic_once', False):
            raise NotImplementedError("geo_ic_once is wired into the multilag model's rollouts only")
        if target_sequence is None: target_sequence = x_sequence
        B,S,_ = target_sequence.size()
        tgt_pw = self.geo_embed.compute_distances_xy_only(target_sequence)
        eh = torch.zeros(self.encoder.num_layers,B,self.encoder.hidden_dim,device=self.device)
        enc_st = (eh, torch.zeros_like(eh))
        z = torch.zeros(B,self.latent_dim,device=self.device)
        noise_state = self.stochastic_head.init_noise_state(B, self.device)
        has_ns = noise_state is not None
        # Decoder AR(1) colored-noise state (zeroed per rollout; None when AR off).
        use_eps = getattr(self.decoder, 'ar1', None) is not None
        eps_prev = torch.zeros(B, self.n_particles*3, device=self.device) if use_eps else None
        cx = x_sequence[:,0,:]
        ce,_ = self.geo_embed(x_sequence[:,0:1,:])
        preds = []
        latent_entropy_sum, latent_top1_sum, latent_step_count = self._init_latent_diagnostics()
        do_ckpt = self.use_gradient_checkpointing and self.training
        for t in range(S-1):
            step_args = (cx, enc_st, z, ce)
            # Slots after curr_emb: ns1, ns2, eps_prev. Pad so each lands right.
            if has_ns:
                step_args = step_args + (noise_state[0], noise_state[1])
            elif use_eps:
                step_args = step_args + (None, None)
            if use_eps:
                step_args = step_args + (eps_prev,)
            if do_ckpt:
                result = ckpt_util.checkpoint(
                    self._forward_step, *step_args, use_reentrant=False)
            else:
                result = self._forward_step(*step_args)
            # eps_t always last (None when AR off).
            if has_ns:
                xp,ds,dm,dsig,enc_st,z,latent_entropy,latent_top1,ns1,ns2,eps_t = result
                noise_state = (ns1, ns2)
            else:
                xp,ds,dm,dsig,enc_st,z,latent_entropy,latent_top1,eps_t = result
            if use_eps:
                eps_prev = eps_t
            latent_entropy_sum, latent_top1_sum, latent_step_count = self._update_latent_diagnostics(
                latent_entropy_sum, latent_top1_sum, latent_step_count, latent_entropy, latent_top1)
            preds.append(xp)
            cx=xp
            ce,_=self.geo_embed(xp.unsqueeze(1))
        xps=torch.stack(preds,dim=1); del preds; A=xps.size(1)
        tp=target_sequence[:,1:A+1]
        puw,tuw = self._unwrap_xy(xps), self._unwrap_xy(tp)
        dl=self.compute_displacement_quantile_loss(xps,tp,pred_xy_unwrapped=puw,target_xy_unwrapped=tuw)
        al=self.compute_angle_quantile_loss(xps,tp,pred_xy_unwrapped=puw,target_xy_unwrapped=tuw) if self.w_angle>0 else None
        vf = self.compute_vacf_quantile_loss(xps,tp,pred_xy_unwrapped=puw,target_xy_unwrapped=tuw)
        alpha_loss = self.compute_alpha_loss(puw, tuw) if self.w_alpha > 0 else None
        vacf_sp = self.compute_vacf_spatial_l1_loss(puw, tuw) if self.w_vacf_spatial > 0 else None
        kurt_loss = self.compute_kurtosis_loss(puw, tuw) if self.w_kurtosis > 0 else None
        del puw,tuw
        ppw=self.geo_embed.compute_distances_xy_only(xps)
        qfn = self._get_quantile_fn()
        tpw=tgt_pw[:,1:A+1]
        pl=qfn(ppw,tpw)
        ph=self.compute_pairwise_tail_loss(ppw,tpw) if self.w_pairwise_tail>0 else None
        ev=self.compute_excluded_volume_loss(ppw,tpw) if self.w_excluded_volume>0 else None
        del ppw,tgt_pw,tpw
        pz=torch.remainder(xps[:,:,:self.n_coords].reshape(B,-1,self.n_particles,3)[:,:,:,2],self.box_size)/self.box_size
        tz=torch.remainder(target_sequence[:,1:A+1,:self.n_coords].reshape(B,-1,self.n_particles,3)[:,:,:,2],self.box_size)/self.box_size
        cl=self.compute_cohesion_loss(xps, tp)
        latent_entropy, latent_top1_prob = self._finalize_latent_diagnostics(
            latent_entropy_sum, latent_top1_sum, latent_step_count)
        out = {
            'displacement_loss':dl,
            'angle_loss':al,
            'pairwise_loss':pl,
            'z_quantile_loss':self.compute_z_quantile_loss(pz,tz),
            'z_displacement_loss':self.compute_z_displacement_quantile_loss(xps,tp),
            'vacf_loss':vf,
            'cohesion_loss':cl,
            'latent_entropy':latent_entropy,
            'latent_top1_prob':latent_top1_prob,
        }
        if al is None:
            del out['angle_loss']
        if ph is not None:
            out['pairwise_tail_loss'] = ph
        if ev is not None:
            out['excluded_volume_loss'] = ev
        if alpha_loss is not None:
            out['alpha_loss'] = alpha_loss
        if vacf_sp is not None:
            out['vacf_spatial_loss'] = vacf_sp
        if kurt_loss is not None:
            out['kurtosis_loss'] = kurt_loss
        return out

    # --- Optimizer ---
    def configure_optimizers(self):
        decay,no_decay = set(),set()
        wl=(nn.Linear,nn.LSTM,nn.GRU,nn.MultiheadAttention); bl=(nn.LayerNorm,nn.GroupNorm,nn.Embedding)
        for mn,m in self.named_modules():
            for pn,p in m.named_parameters():
                fpn=f'{mn}.{pn}' if mn else pn
                if pn.endswith('bias'): no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m,wl): decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m,bl): no_decay.add(fpn)
        pd={pn:p for pn,p in self.named_parameters()}; assert not(decay&no_decay)
        no_decay.update(set(pd.keys())-decay-no_decay)
        wd=self.hparams.get('weight_decay',1e-2); lr=self.hparams.get('learning_rate',1e-4)
        dyn_lr_mult = self.hparams.get('dynamics_lr_mult', 1.0)
        dyn_names = {n for n in pd if '_raw_decay' in n or '_raw_innov_scale' in n
                     or '_log_theta' in n or '_log_dt' in n
                     or 'noise_filter.' in n or '.ar1.' in n}
        # The squash head sits behind a sigmoid biased to start at identity, so its
        # gradient is throttled by sigmoid'(init_bias) -- 0.0025 at the default bias
        # of 6.0, ~100x below the sigmoid's peak slope. Trained standalone it needed
        # lr~1e-2 while the backbone wants ~1e-4, so it gets its own group. Left in
        # the backbone's group it simply does not move, and the run looks normal.
        sq_lr_mult = self.hparams.get('squash_lr_mult', 1.0)
        sq_names = {n for n in pd if n.startswith('squash_head.')}
        held = dyn_names | sq_names
        og=[{"params":[pd[n] for n in sorted(decay) if n not in held],"weight_decay":wd},
            {"params":[pd[n] for n in sorted(no_decay) if n not in held],"weight_decay":0.0}]
        if dyn_names:
            og.append({"params":[pd[n] for n in sorted(dyn_names)],
                        "weight_decay":0.0, "lr": lr * dyn_lr_mult})
        if sq_names:
            og.append({"params":[pd[n] for n in sorted(sq_names)],
                        "weight_decay":0.0, "lr": lr * sq_lr_mult})
        if self.hparams.get('optimizer_type', 'adamw') == 'sophia_g':
            from nbcore.sophia_g import SophiaG
            opt=SophiaG(params=og,lr=lr,
                        rho=self.hparams.get('sophia_rho',0.04),
                        bs=self.hparams.get('batch_size',512)*self.hparams.get('accumulate_grad_batches',1),
                        weight_decay=wd)
        else:
            opt=torch.optim.AdamW(og,lr=lr,weight_decay=wd, betas=(0.9, 0.95))
        # lr_schedule was live in diffusion/base_model.py only; this copy hardcoded
        # ReduceLROnPlateau, so setting lr_schedule in a MEMBRANE config was a SILENT
        # NO-OP -- accepted by the key validator, read by nothing. Exactly the failure
        # CLAUDE.md records for the 30-cell seed sweep. 'plateau' reproduces the old
        # behaviour exactly, so configs that do not set lr_schedule are unchanged.
        from nbcore.lr_schedule import build_scheduler_config
        return {"optimizer": opt,
                "lr_scheduler": build_scheduler_config(opt, self.hparams)}

    # --- Training step ---
    def training_step(self, batch, batch_idx):
        multi_sys = self.hparams.get('multi_system', False)
        membrane = getattr(self, 'membrane_mode', False)
        membrane_hg = getattr(self, 'membrane_hypergraph', False)
        start_frame_idx = None
        mem_node_gt = mem_hyper_gt = theta_gt = blade_gt = None
        if multi_sys and isinstance(batch, (list, tuple)) and len(batch) == 5 and not membrane_hg:
            inp, active_mask, edge_src, edge_dst, edge_mask = batch
            target = inp
        elif membrane_hg and isinstance(batch, (list, tuple)) and len(batch) in (3, 4, 5):
            # (coords[B,S,21], node[B,S,7], hyper[B,S,120], [theta[B,S,7] when arms
            # on], [blade[B,S,7,3,3] when geom_blade_enabled])
            inp, mem_node_gt, mem_hyper_gt = batch[0], batch[1], batch[2]
            theta_gt = batch[3] if len(batch) >= 4 else None
            blade_gt = batch[4] if len(batch) == 5 else None
            target = inp
            active_mask = edge_src = edge_dst = edge_mask = None
        elif membrane and isinstance(batch, (list, tuple)) and len(batch) == 2:
            inp, start_frame_idx = batch          # (window [B,S,21], start_idx [B])
            target = inp
            active_mask = edge_src = edge_dst = edge_mask = None
        else:
            if isinstance(batch,(list,tuple)): batch=batch[0]
            inp = batch
            target = batch
            active_mask = edge_src = edge_dst = edge_mask = None
        K = self.hparams.get('n_rollout_samples', 1)
        if K > 1:
            inp = inp.repeat_interleave(K, dim=0)
            target = target.repeat_interleave(K, dim=0)
            if start_frame_idx is not None:
                start_frame_idx = start_frame_idx.repeat_interleave(K, dim=0)
            if mem_node_gt is not None:
                mem_node_gt = mem_node_gt.repeat_interleave(K, dim=0)
                mem_hyper_gt = mem_hyper_gt.repeat_interleave(K, dim=0)
            if theta_gt is not None:
                theta_gt = theta_gt.repeat_interleave(K, dim=0)
            if blade_gt is not None:
                blade_gt = blade_gt.repeat_interleave(K, dim=0)
            if active_mask is not None:
                active_mask = active_mask.repeat_interleave(K, dim=0)
                edge_src = edge_src.repeat_interleave(K, dim=0)
                edge_dst = edge_dst.repeat_interleave(K, dim=0)
                edge_mask = edge_mask.repeat_interleave(K, dim=0)
        ns=self.hparams.get('input_noise_sigma',0.0)
        inp=torch.remainder(inp+torch.randn_like(inp)*ns,self.box_size) if ns>0 else inp
        _fwd=dict(x_sequence=inp, target_sequence=target, active_mask=active_mask,
                  edge_src=edge_src, edge_dst=edge_dst, edge_mask=edge_mask)
        if start_frame_idx is not None:
            _fwd['start_frame_idx'] = start_frame_idx
        if theta_gt is not None:
            _fwd['theta_gt'] = theta_gt
        if blade_gt is not None:
            _fwd['blade'] = blade_gt[:, 0]     # IC frame; the AR loop advances it from here
            # Full GT blade sequence, for the blade-sequence losses -- gated on the
            # weights (not just their downstream use) by the same helper
            # model_multilag.forward's capture_blades uses for the PREDICTED sequence:
            # otherwise every arms_enabled+advances_blade run threads a [B,S,P,3,3]
            # tensor into forward() that no loss reads.
            if self.rotation_losses_enabled():
                _fwd['blade_gt'] = blade_gt
        ld=self.forward(**_fwd)
        # pred_blades [B,A,P,3,3] is forward()'s intermediate for building the
        # blade-sequence losses (mirrors the 1-DOF path's own theta_ps, which never
        # leaves forward() at all) -- not a scalar, so it must not reach log_dict
        # below. Only present when capture_blades gated it in
        # (rotation_losses_enabled()), i.e. exactly when this would otherwise crash
        # the log call.
        ld.pop('pred_blades', None)
        # Same treatment for the position tensors the coupling loss needs: forward
        # only exports them when w_mem_coupling>0, and they are [B,A,D], not scalars.
        pred_positions = ld.pop('pred_positions', None)
        target_positions = ld.pop('target_positions', None)
        # Membrane-hypergraph aux loss: predicted node/hyperedge depths vs GT.
        # Pop the [B,A,*] prediction tensors so they aren't fed to scalar logging.
        if membrane_hg and 'mem_node_pred' in ld:
            A = ld['mem_node_pred'].size(1)
            mnp = ld.pop('mem_node_pred')            # [B,A,7]
            mhp = ld.pop('mem_hyper_pred')           # [B,A,120]
            mni = ld.pop('mem_node_init')            # [B,7]
            mhi = ld.pop('mem_hyper_init')           # [B,120]
            node_gt = mem_node_gt[:, 1:A + 1]        # [B,A,7]
            hyper_gt = mem_hyper_gt[:, 1:A + 1]      # [B,A,120]
            node_gt0, hyper_gt0 = mem_node_gt[:, 0], mem_hyper_gt[:, 0]
            # membrane_loss_units='delta' rescales prediction AND GT by the same
            # per-target factor, so it changes the UNITS the terms below are computed
            # in, never the comparison. Which terms that actually moves:
            #   membrane_loss   MSE            -> s^2  (x237 node, x88.5 hyper)
            #   mem_value       sort-match L1  -> s
            #   mem_step        |Δd| L1        -> s
            #   mem_pairwise    |d_i-d_j| L1   -> s
            #   mem_cohesion    Δ|d_i-d_j| L1  -> s
            #   mem_autocorr    c / var        -> INVARIANT. _acurve divides the
            #     lagged product of Δdepth by the variance of that same Δdepth, so a
            #     common factor cancels top and bottom. It gets the UNSCALED tensors:
            #     that makes the invariance explicit, and keeps the term bit-identical
            #     across the two modes instead of jittering by float rounding.
            # mem_consistency is also left alone -- its slope (box/node_std) is a
            # fixed contract with the decoder's z, and scaling one side would break it.
            if self.mem_loss_units == 'delta':
                sn, sh = self.mem_loss_scale_node, self.mem_loss_scale_hyper
                mnp_s, mni_s = mnp * sn, mni * sn
                node_gt_s, node_gt0_s = node_gt * sn, node_gt0 * sn
                mhp_s, mhi_s = mhp * sh, mhi * sh
                hyper_gt_s, hyper_gt0_s = hyper_gt * sh, hyper_gt0 * sh
            else:
                mnp_s, mni_s, node_gt_s, node_gt0_s = mnp, mni, node_gt, node_gt0
                mhp_s, mhi_s, hyper_gt_s, hyper_gt0_s = mhp, mhi, hyper_gt, hyper_gt0
            # Pointwise MSE aux — kept only if w_membrane>0 (dropped in the fully
            # distributional variant by setting w_membrane=0).
            #
            # ONE MSE PER FIELD OVER THE WHOLE SEQUENCE, init concatenated as frame 0.
            # This used to be FOUR separately mean-reduced terms summed with equal
            # coefficients -- rollout node, rollout hyper, init node, init hyper -- and
            # F.mse_loss reduces to a mean, so the two init terms each carried the same
            # weight as an entire A-frame rollout. Measured on real windows: the single
            # init frame owned 48% of membrane_loss at seq_len 3000 (35.1% node +
            # 12.9% hyper) while being 1/3000 of the supervised frames, i.e. 1440x
            # over-weighted per frame; 50% / 64x at seq_len 128. Every membrane run
            # before 2026-09-13 trained under that, which is part of why depth came out
            # variance-collapsed: half the gradient was fitting one frame the module
            # sees GT for anyway.
            #
            # Concatenating instead of re-weighting keeps per-frame weight uniform by
            # construction and cannot drift if A changes. NOTE THE SCALE CHANGE: the
            # term is now ~0.52x its old value at seq_len 3000 (the init MSEs no longer
            # contribute half the sum), so a w_membrane copied from a pre-fix config is
            # effectively ~2x larger than it was. The distributional mem_* terms are
            # deliberately NOT changed -- they never included the init frame, and adding
            # 1 frame in 3000 to a sorted-quantile match would move nothing while
            # invalidating the study in flight.
            if self.w_membrane > 0:
                mn_p = torch.cat([mni_s.unsqueeze(1), mnp_s], dim=1)        # [B,A+1,7]
                mn_g = torch.cat([node_gt0_s.unsqueeze(1), node_gt_s], dim=1)
                mh_p = torch.cat([mhi_s.unsqueeze(1), mhp_s], dim=1)        # [B,A+1,120]
                mh_g = torch.cat([hyper_gt0_s.unsqueeze(1), hyper_gt_s], dim=1)
                ld['membrane_loss'] = F.mse_loss(mn_p, mn_g) + F.mse_loss(mh_p, mh_g)
            if self.w_mem_init > 0:
                ld['mem_init_loss'] = F.mse_loss(mni_s, node_gt0_s) + F.mse_loss(mhi_s, hyper_gt0_s)
            # Distributional depth losses (analogs of the coordinate quantile losses).
            if self.w_mem_value > 0:
                ld['mem_value_loss'] = (self.compute_quantile_loss_per_particle(mnp_s, node_gt_s)
                                        + self.compute_quantile_loss_pooled(mhp_s, hyper_gt_s))
            if self.w_mem_step > 0:
                ld['mem_step_loss'] = (self._mem_step_qloss(mnp_s, node_gt_s)
                                       + self._mem_step_qloss(mhp_s, hyper_gt_s))
            if self.w_mem_autocorr > 0:
                ld['mem_autocorr_loss'] = (self._mem_autocorr_loss(mnp, node_gt)
                                           + self._mem_autocorr_loss(mhp, hyper_gt))
            if self.w_mem_pairwise > 0:
                ld['mem_pairwise_loss'] = self._mem_pairwise_qloss(mnp_s, node_gt_s)
            if self.w_mem_cohesion > 0:
                ld['mem_cohesion_loss'] = self._mem_cohesion_qloss(mnp_s, node_gt_s)
            if self.w_mem_coupling > 0 and pred_positions is not None:
                ld['mem_coupling_loss'] = self._mem_coupling_qloss(
                    mnp, node_gt, pred_positions, target_positions)
        w={'displacement_loss':self.w_displacement*ld['displacement_loss']}
        # a zero-weight angle term is not computed at all
        if self.w_angle > 0 and 'angle_loss' in ld:
            w['angle_loss'] = self.w_angle*ld['angle_loss']
        w['pairwise_loss'] = self.w_pairwise*ld['pairwise_loss']
        if self.predict_z:
            w['z_quantile_loss'] = self.w_z_quantile*ld['z_quantile_loss']
        w['vacf_loss'] = self.w_vacf*ld['vacf_loss']
        w['cohesion_loss'] = self.w_cohesion*ld['cohesion_loss']
        if self.w_logpair_kl > 0:
            w['logpair_kl_loss'] = self.w_logpair_kl * ld['logpair_kl_loss']
        if self.w_dlogpair_kl > 0:
            w['dlogpair_kl_loss'] = self.w_dlogpair_kl * ld['dlogpair_kl_loss']
        if self.w_logpair_q > 0:
            w['logpair_q_loss'] = self.w_logpair_q * ld['logpair_q_loss']
        if self.w_dlogpair_q > 0:
            w['dlogpair_q_loss'] = self.w_dlogpair_q * ld['dlogpair_q_loss']
        if self.predict_z:
            w['z_displacement_loss'] = self.w_z_displacement*ld['z_displacement_loss']
        # Optional lag-wise mean-VACF MSE term. Off by default (w_vacf_mse=0).
        if self.w_vacf_mse > 0 and 'vacf_mse_loss' in ld:
            w['vacf_mse_loss'] = self.w_vacf_mse * ld['vacf_mse_loss']
        # Optional latent-velocity autocorrelation term. Off by default.
        if self.w_latent_corr > 0 and 'latent_corr_loss' in ld:
            w['latent_corr_loss'] = self.w_latent_corr * ld['latent_corr_loss']
        if self.w_pairwise_tail > 0 and 'pairwise_tail_loss' in ld:
            w['pairwise_tail_loss'] = self.w_pairwise_tail * ld['pairwise_tail_loss']
        if self.w_squash > 0 and 'squash_loss' in ld:
            w['squash_loss'] = self.w_squash * ld['squash_loss']
        if self.w_excluded_volume > 0 and 'excluded_volume_loss' in ld:
            # ev_w_frac_cur is 1.0 unless ev_w_schedule is set, and x*1.0 is exact.
            w['excluded_volume_loss'] = (self.w_excluded_volume * self.ev_w_frac_cur
                                         * ld['excluded_volume_loss'])
        if getattr(self, 'arms_enabled', False):
            if self.w_arm_pairwise > 0 and 'arm_pairwise_loss' in ld:
                w['arm_pairwise_loss'] = self.w_arm_pairwise * ld['arm_pairwise_loss']
            # theta_step_loss comes from two different places, deliberately keyed the
            # same because it is the same observable: the 1-DOF arms difference the
            # rotation head's own accumulated theta, the 3-DOF arms difference
            # blade_theta() of the predicted blade state. One weight, one logged key,
            # so the two arm families' spin supervision is directly comparable.
            if self.w_theta_step > 0 and 'theta_step_loss' in ld:
                w['theta_step_loss'] = self.w_theta_step * ld['theta_step_loss']
            # rot_step_loss/tilt_loss are only populated when forward() both stacks a
            # predicted blade sequence (model_multilag.forward's out['pred_blades'],
            # [B,A,P,3,3]) AND was handed the GT blade sequence to compare it to
            # (this training_step's blade_gt, threaded through as forward()'s
            # blade_gt kwarg above, gated by the same rotation_losses_enabled()). A
            # non-zero weight below with the key absent from ld would be a bug, not
            # the expected steady state -- and the construction-time guard in
            # _init_config now rejects the one config that used to produce it (these
            # weights on an arm with advances_blade=False) -- so 'in ld' is kept only
            # as a defensive guard.
            if self.w_rot_step > 0 and 'rot_step_loss' in ld:
                w['rot_step_loss'] = self.w_rot_step * ld['rot_step_loss']
            if self.w_tilt > 0 and 'tilt_loss' in ld:
                w['tilt_loss'] = self.w_tilt * ld['tilt_loss']
            if self.w_arm_excluded_volume > 0 and 'arm_excluded_volume_loss' in ld:
                w['arm_excluded_volume_loss'] = self.w_arm_excluded_volume * ld['arm_excluded_volume_loss']
        if self.w_alpha > 0 and 'alpha_loss' in ld:
            w['alpha_loss'] = self.w_alpha * ld['alpha_loss']
        if self.w_vacf_spatial > 0 and 'vacf_spatial_loss' in ld:
            w['vacf_spatial_loss'] = self.w_vacf_spatial * ld['vacf_spatial_loss']
        # Optional NGP/kurtosis matching term. Off by default (w_kurtosis=0).
        if self.w_kurtosis > 0 and 'kurtosis_loss' in ld:
            w['kurtosis_loss'] = self.w_kurtosis * ld['kurtosis_loss']
        if self.w_membrane > 0 and 'membrane_loss' in ld:
            w['membrane_loss'] = self.w_membrane * ld['membrane_loss']
        if self.w_mem_init > 0 and 'mem_init_loss' in ld:
            w['mem_init_loss'] = self.w_mem_init * ld['mem_init_loss']
        if self.w_mem_value > 0 and 'mem_value_loss' in ld:
            w['mem_value_loss'] = self.w_mem_value * ld['mem_value_loss']
        if self.w_mem_step > 0 and 'mem_step_loss' in ld:
            w['mem_step_loss'] = self.w_mem_step * ld['mem_step_loss']
        if self.w_mem_autocorr > 0 and 'mem_autocorr_loss' in ld:
            w['mem_autocorr_loss'] = self.w_mem_autocorr * ld['mem_autocorr_loss']
        if self.w_mem_pairwise > 0 and 'mem_pairwise_loss' in ld:
            w['mem_pairwise_loss'] = self.w_mem_pairwise * ld['mem_pairwise_loss']
        if self.w_mem_cohesion > 0 and 'mem_cohesion_loss' in ld:
            w['mem_cohesion_loss'] = self.w_mem_cohesion * ld['mem_cohesion_loss']
        if self.w_mem_coupling > 0 and 'mem_coupling_loss' in ld:
            w['mem_coupling_loss'] = self.w_mem_coupling * ld['mem_coupling_loss']
        if self.w_mem_consistency > 0 and 'mem_consistency' in ld:
            w['mem_consistency'] = self.w_mem_consistency * ld['mem_consistency']
        total=sum(w.values())
        if os.environ.get('MEMBRANE_LOSS_DEBUG') and batch_idx == 0:
            br={k:round(float(v),4) for k,v in sorted(w.items(),key=lambda kv:-float(kv[1]))}
            rw={k:round(float(v),5) for k,v in ld.items() if k.endswith('_loss')}
            print(f"\n[loss-debug] total={float(total):.3f}\n  weighted (desc): {br}\n  raw: {rw}", flush=True)
        if torch.isnan(total):
            nc={k:v.item() for k,v in w.items() if torch.isnan(v)}
            print(f"\n{'='*60}\nNaN (batch={batch_idx}, epoch={self.current_epoch})\n{nc}\n{'='*60}")
            traceback.print_stack(); return None
        self.log_dict({f"train/{k}":v for k,v in ld.items()},on_step=True,on_epoch=True)
        self.log_dict({f"weighted/{k}":v for k,v in w.items()},on_step=True,on_epoch=True)
        self.log("train_loss",total,prog_bar=True,on_step=False,on_epoch=True)
        self.log("seq_len_actual",float(inp.size(1)),on_step=False,on_epoch=True)
        # Only logged when a ramp is configured, so the logged-key set of every
        # pre-existing run stays exactly as it was.
        if self.ev_w_schedule != 'constant':
            self.log("train/ev_w_frac", self.ev_w_frac_cur, on_step=False, on_epoch=True)
            self.log("train/ev_force_cur", float(self.ev_force_cur),
                     on_step=False, on_epoch=True)
        if hasattr(self.stochastic_head, 'effective_theta'):
            self.log("train/theta", float(self.stochastic_head.effective_theta),
                     on_step=True, on_epoch=True)
        if hasattr(self.stochastic_head, 'effective_dt'):
            eff_dt = float(self.stochastic_head.effective_dt)
            self.log("train/dt", eff_dt, on_step=True, on_epoch=True)
            self.log("train/theta_dt", float(self.stochastic_head.effective_theta) * eff_dt,
                     on_step=True, on_epoch=True)
        if hasattr(self.stochastic_head, 'effective_decay'):
            self.log("train/decay", float(self.stochastic_head.effective_decay),
                     on_step=True, on_epoch=True)
        if hasattr(self.stochastic_head, 'effective_innov_scale'):
            self.log("train/innov_scale", float(self.stochastic_head.effective_innov_scale),
                     on_step=True, on_epoch=True)
        if getattr(self.stochastic_head, 'noise_filter', None) is not None:
            self.log("train/noise_corr", float(self.stochastic_head.noise_filter.effective_noise_corr),
                     on_step=True, on_epoch=True)
        return total

    # --- Inference ---
    def predict_future(self, initial_x, steps, return_unwrapped=False):
        if self.hparams.get('geo_ic_once', False):
            raise NotImplementedError("geo_ic_once is wired into the multilag model's rollouts only")
        B=initial_x.size(0)
        eh=torch.zeros(self.encoder.num_layers,B,self.encoder.hidden_dim,device=self.device)
        enc_st=(eh,torch.zeros_like(eh)); z=torch.zeros(B,self.latent_dim,device=self.device)
        cx=initial_x[:,0,:]; cu=cx.clone(); ps,pu=[],[]
        # Decoder AR(1) state threaded internally so eval matches training.
        use_eps = getattr(self.decoder, 'ar1', None) is not None
        eps_prev = torch.zeros(B, self.n_particles*3, device=self.device) if use_eps else None
        for _ in range(steps):
            ce,_=self.geo_embed(cx.unsqueeze(1))
            transition_params,enc_st=self.encoder(ce,enc_st)
            transition_params = squeeze_transition_params(transition_params, dim=1)
            z,_,_,_=self.stochastic_head.step(transition_params,z)
            ds,_,_,eps_t=self.decoder(z,ce,eps_prev=eps_prev)
            if use_eps:
                eps_prev = eps_t
            nu=cu+ds; nw=torch.remainder(nu,self.box_size)
            ps.append(nw); pu.append(nu); cx,cu=nw,nu
        sw=torch.stack(ps,dim=1)
        return (sw,torch.stack(pu,dim=1)) if return_unwrapped else sw
