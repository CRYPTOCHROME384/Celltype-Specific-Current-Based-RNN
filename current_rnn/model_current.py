# current_rnn/model_current.py

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ALMCurrentRNN(nn.Module):
    """
    High-dimensional current-based RNN for ALM data.

    - State h_t ∈ R^N corresponds 1:1 to recorded neurons (after cell-type filtering).
    - Discrete-time dynamics (Euler):

        h_{t+1} = h_t + (dt / tau) * [ -h_t + J * phi(h_t) + W_in * u_t + b ]

      where:
        - J: NxN recurrent coupling
        - W_in: NxD_in input weight
        - u_t: external input at time t, shape [B, D_in]
        - phi: element-wise nonlinearity

    - Forward input u: shape [B, T, D_in]
    - Forward output: dict with
        - "h":    shape [B, T, N] (current / latent state)
        - "rate": shape [B, T, N] (phi(h))
    """

    def __init__(
        self,
        N: int,
        D_in: int,
        dt: float = 1.0,
        tau: float = 1.0,
        substeps: int = 1,
        nonlinearity: str = "tanh",
        device: Optional[torch.device] = None,
        dale_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            N:        number of neurons (after cell-type filtering)
            D_in:     dimension of external input u_t
            dt:       integration time step (in arbitrary units, typically 1 frame)
            tau:      effective time constant of the dynamics
            nonlinearity: 'tanh', 'relu', or 'softplus'
            device:   torch device; if None, inferred from parameters at runtime
            dale_mask: optional tensor of shape [N, N] with entries in {-1, 0, +1}
                       to encode Dale's law constraints (not enforced yet, but kept
                       for future use). For now we only store it.
        """
        super().__init__()

        self.N = N
        self.D_in = D_in
        self.dt = float(dt)
        self.tau = float(tau)
        self.substeps = int(substeps)
        assert self.substeps >= 1
        self.nonlinearity = nonlinearity.lower()

        # ---------------------------------------------------------------------
        # Recurrent matrix J ∈ R^{N×N}
        # ---------------------------------------------------------------------
        # Xavier-like initialization, scaled by 1/sqrt(N) to avoid instabilities.
        J = torch.empty(N, N)
        nn.init.kaiming_uniform_(J, a=math.sqrt(5))
        J = J / math.sqrt(N)
        self.J = nn.Parameter(J)

        # ---------------------------------------------------------------------
        # Input weights W_in ∈ R^{N×D_in}
        # ---------------------------------------------------------------------
        W_in = torch.empty(N, D_in)
        nn.init.kaiming_uniform_(W_in, a=math.sqrt(5))
        W_in = W_in / math.sqrt(D_in)
        self.W_in = nn.Parameter(W_in)

        # ---------------------------------------------------------------------
        # Bias term b ∈ R^N
        # ---------------------------------------------------------------------
        self.b = nn.Parameter(torch.zeros(N))

        # Optional Dale's law mask: stored but not enforced yet.
        # Expected shape: [N, N], entries in {-1, 0, +1}
        if dale_mask is not None:
            if dale_mask.shape != (N, N):
                raise ValueError(
                    f"dale_mask must have shape ({N}, {N}), "
                    f"got {tuple(dale_mask.shape)}"
                )
            self.register_buffer("dale_mask", dale_mask.clone())
        else:
            self.dale_mask = None

        # For consistent dtype/device handling later
        if device is not None:
            self.to(device)

    # -------------------------------------------------------------------------
    # Nonlinearity
    # -------------------------------------------------------------------------
    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlinearity == "tanh":
            return torch.tanh(x)
        elif self.nonlinearity == "relu":
            return F.relu(x)
        elif self.nonlinearity == "softplus":
            return F.softplus(x)
        else:
            raise ValueError(f"Unsupported nonlinearity: {self.nonlinearity}")

    # -------------------------------------------------------------------------
    # Optional: apply Dale's law mask to J (for future use)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def apply_dale_mask(self) -> None:
        """
        Enforce Dale's law sign structure on J if a mask is provided.

        dale_mask[i, j] = +1  -> J[i, j] constrained to be ≥ 0
        dale_mask[i, j] = -1  -> J[i, j] constrained to be ≤ 0
        dale_mask[i, j] =  0  -> unconstrained

        For now, we simply project J onto the sign cone defined by dale_mask.
        This can be called manually after optimizer.step() in a training loop.
        """
        if self.dale_mask is None:
            return

        # Positive-constrained entries
        pos_mask = self.dale_mask > 0
        # Negative-constrained entries
        neg_mask = self.dale_mask < 0

        with torch.no_grad():
            self.J.data[pos_mask] = torch.clamp_min(self.J.data[pos_mask], 0.0)
            self.J.data[neg_mask] = torch.clamp_max(self.J.data[neg_mask], 0.0)

    # -------------------------------------------------------------------------
    # Forward dynamics
    # -------------------------------------------------------------------------
    def forward(
    self,
    u: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
    return_rate: bool = True,
) -> Dict[str, torch.Tensor]:
        """
        Run the RNN forward for a batch of input trajectories.

     Args:
        u:  external input trajectories, shape [B, T, D_in]
        h0: optional initial state, shape [B, N]. If None, initialized to zeros.
        noise_std: standard deviation of additive Gaussian noise on h.
        return_rate: if True, also return phi(h) as 'rate'.

        Returns:
        A dict with:
            'h':    tensor [B, T, N], the current-based state
            'rate': tensor [B, T, N], phi(h)       (if return_rate=True)
        """
        if u.dim() != 3:
            raise ValueError(f"u must have shape [B, T, D_in], got {tuple(u.shape)}")

        B, T, D_in = u.shape
        if D_in != self.D_in:
            raise ValueError(f"Expected input dimension D_in={self.D_in}, got {D_in}")

        device = self.J.device
        u = u.to(device)

        # Initialize h_0
        if h0 is None:
            h_t = torch.zeros(B, self.N, device=device, dtype=self.J.dtype)
        else:
            if h0.shape != (B, self.N):
                raise ValueError(
                    f"h0 must have shape [B, N]=[{B}, {self.N}], got {tuple(h0.shape)}"
                )
            h_t = h0.to(device)
        # -------- sub-stepping setup --------
        substeps = int(getattr(self, "substeps", 1))
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")

        dt_sub = self.dt / substeps
        dt_over_tau_sub = dt_sub / self.tau  # = (self.dt/self.tau)/substeps

        # If you want the total noise variance per *frame* to stay comparable
        # when using multiple substeps, scale per-substep noise by 1/sqrt(substeps).
        sqrt_dt_sub = math.sqrt(dt_sub)
        noise_scale = 1.0 / math.sqrt(substeps) if substeps > 1 else 1.0

        h_seq = []
        for t in range(T):
            # u_t: [B, D_in] (kept constant within substeps)
            u_t = u[:, t, :]

            # Integrate substeps times within this frame
            for _ in range(substeps):
                # rate_t = phi(h_t)  -> [B, N]
                rate_t = self._phi(h_t)

                # recurrent input: [B, N]
                rec_t = torch.matmul(rate_t, self.J.T)

                # external input: [B, N]
                inp_t = torch.matmul(u_t, self.W_in.T)

                # drift: dh/dt = -h + rec + inp + b
                drift = -h_t + rec_t + inp_t + self.b

                # Euler substep
                h_t = h_t + dt_over_tau_sub * drift

                # Noise (optional)
                if noise_std > 0.0:
                    # per-substep noise; noise_std interpreted as per-frame scale
                    noise = (noise_std * noise_scale) * sqrt_dt_sub * torch.randn_like(h_t)
                    h_t = h_t + noise
            # record state at frame boundary
            h_seq.append(h_t)

        # Stack over time: list of [B, N] -> [B, T, N]
        h_seq = torch.stack(h_seq, dim=1)  # [B, T, N]

        out = {"h": h_seq}
        if return_rate:
            out["rate"] = self._phi(h_seq)

        return out
    
