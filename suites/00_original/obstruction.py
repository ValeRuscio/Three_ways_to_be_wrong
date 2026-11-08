"""Pinned delivery residual R(S) for the prompt-conditioned affine system.

Relative extension problem (Sec. 'obstruction' of the plan):
  unknowns : residual states on the causal cone of the source span
             (layers 1..L, positions >= min(S)), EXCEPT pinned nodes;
  pins     : layer 0 everywhere; positions < min(S) at all layers (satisfied
             automatically by causality); the source span S for layers
             1..pin_L  (this is the pinned source state);
  demand   : success-level target delivery at the final-position state,
             <u_g, x_hat_{L,T}> = c_plus, weighted by lam.

The realized trajectory satisfies every dynamics row exactly (conservation /
reconstruction certificates), so at delta = 0 the only nonzero residual is the
terminal delivery row.  R(S) is therefore the minimal frozen-dynamics defect
energy required to extend the pinned source section to a target-delivering
terminal state:

  R(S)^2 = min_{delta on free nodes} sum_l ||X_{l+1} - F_l(X_l)||^2
            + lam^2 (<u_g, x_hat_{L,T}(X)> - c_plus)^2 ,   X = realized + delta.

R(S) ~ 0        <=> the pinned extension exists (delivery is reachable inside
                     the selected affine without breaking the dynamics);
R(S) large      <=> obstruction: no such extension (transport failure), or the
                     section itself carries no target (source failure).
The residual DEPTH PROFILE separates those two: source -> energy at the
source/early layers; transport -> energy along the route.

Solved matrix-free by CG on the normal equations.  The adjoint is exact
because the frozen map is linear (autograd vjp == transpose); certificate C3
checks R is affine to numerical precision.
"""
from dataclasses import dataclass
from typing import List, Sequence
import torch

from frozen_cache import Weights, ForwardCache, frozen_layer, apply_norm


@dataclass
class ObstructionResult:
    ob: float                 # sqrt(dynamics residual energy) at optimum
    ob_total: float           # including the terminal penalty row
    delivered_gap0: float     # c_plus - realized delivery (before solving)
    delivered_gap: float      # residual delivery gap at optimum
    profile: torch.Tensor     # [L] per-layer dynamics residual energy
    depth_centroid: float     # energy-weighted mean relative depth
    pi_S: float               # source patch content, for normalization
    ob_norm: float            # ob / max(pi_S, eps)
    cg_iters: int
    cert_affine: float        # C3: affinity check of the residual operator


def _free_mask(L: int, T: int, d: int, S: Sequence[int], pin_L: int,
               device) -> torch.Tensor:
    """[L, T, d] mask of free coordinates for X_1..X_L (layer l stored at l-1)."""
    m = torch.zeros(L, T, 1, device=device)
    i0 = min(S)
    m[:, i0:, :] = 1.0
    for i in S:
        m[:pin_L, i, :] = 0.0                 # pinned source section
    return m.expand(L, T, d).contiguous()


def solve_obstruction(W: Weights, C: ForwardCache, S: Sequence[int],
                      tok_g: int, c_plus: float, pin_L: int = None,
                      lam: float = 30.0, cg_iters: int = 150,
                      cg_tol: float = 1e-6, mu: float = 1e-6,
                      mid_band=(0.25, 0.75),
                      cert_tol: float = 1e-3) -> ObstructionResult:
    assert list(S) == list(range(min(S), max(S) + 1)), \
        "source span must be contiguous (cone bookkeeping assumes it)"
    L, T, d = W.L, C.resid[0].shape[0], C.resid[0].shape[1]
    dev = C.resid[0].device
    pin_L = pin_L if pin_L is not None else L // 2
    realized = torch.stack(C.resid[1:])                     # [L, T, d]
    mask = _free_mask(L, T, d, S, pin_L, dev)
    ug_hat = (C.inv_f[-1] * W.ln_f) * W.WU[tok_g]           # frozen-lens row, [d]

    def delivered(xrow):
        """<u_g, x_hat> with frozen final inv; center-aware (LayerNorm)."""
        xh = apply_norm(xrow.view(1, -1), C.inv_f[-1:], W.ln_f, W.center)[0]
        return (xh * W.WU[tok_g]).sum()

    # One-sided demand: only a delivery SHORTFALL is an obstruction.  If the
    # realized delivery already meets c_plus, the pinned extension is the
    # realized section itself and R(S) = 0 exactly.
    a0 = float(delivered(realized[-1][-1]))
    if a0 >= c_plus:
        lo, hi = int(mid_band[0] * L), int(mid_band[1] * L)
        pi = max(float((C.resid[l][j] * ug_hat.squeeze()).sum())
                 for l in range(lo, hi + 1) for j in S)
        return ObstructionResult(
            ob=0.0, ob_total=0.0, delivered_gap0=a0 - c_plus,  # >0: surplus
            delivered_gap=0.0, profile=torch.zeros(L), depth_centroid=0.0,
            pi_S=pi, ob_norm=0.0, cg_iters=0, cert_affine=0.0)

    def residuals(delta: torch.Tensor):
        X = realized + delta * mask
        dyn = []
        x_prev = C.resid[0]                                 # layer 0 pinned
        for l in range(L):
            pred = frozen_layer(x_prev, W, l, C)
            dyn.append(X[l] - pred)      # off-cone rows vanish identically (causality)
            x_prev = X[l]
        dyn = torch.stack(dyn)                              # [L, T, d]
        delivery = lam * (delivered(X[-1][-1]) - c_plus)
        return dyn, delivery

    # --- affine structure: R(delta) = R0 + A delta -------------------------
    zero = torch.zeros_like(realized)
    with torch.no_grad():
        dyn0, del0 = residuals(zero)

    def A_apply(delta):
        with torch.no_grad():
            dyn, dl = residuals(delta)
        return dyn - dyn0, dl - del0

    def A_adjoint(y_dyn, y_del):
        # exact adjoint via autograd on the (linear) residual operator;
        # enable_grad shields against a globally disabled autograd state
        with torch.enable_grad():
            delta = torch.zeros_like(realized, requires_grad=True)
            dyn, dl = residuals(delta)
            s = (dyn * y_dyn).sum() + dl * y_del
            (g,) = torch.autograd.grad(s, delta)
        return g

    # Certificate C3: R must be affine (linearity of the frozen map).
    with torch.no_grad():
        probe = torch.randn_like(realized) * mask
        d1, l1 = A_apply(probe)
        d2, l2 = A_apply(2 * probe)
        cert = float(((d2 - 2 * d1).norm() + (l2 - 2 * l1).abs()) /
                     (d1.norm() + l1.abs() + 1e-30))
    # Structural linearity is exact; the tolerance absorbs fp32 cancellation
    # in R(delta) - R0 (dynamics rows are ~0 at the realized trajectory).
    assert cert < cert_tol, \
        f"C3 failed: residual operator not affine ({cert:.2e})"

    # --- CG on (A^T A + mu I) delta = -A^T R0 ------------------------------
    b = -A_adjoint(dyn0, del0)
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = (r * r).sum()
    it = 0
    for it in range(1, cg_iters + 1):
        Ad, Al = A_apply(p)
        Ap = A_adjoint(Ad, Al) + mu * p
        pAp = (p * Ap).sum()
        if not torch.isfinite(pAp) or pAp <= 0:
            break                       # degenerate direction: keep best x
        alpha = rs / pAp
        x_new = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if not (torch.isfinite(rs_new) and torch.isfinite(x_new).all()):
            break                       # numerical breakdown: keep best x
        x = x_new
        if rs_new.sqrt() < cg_tol * b.norm() or rs_new < 1e-30:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new

    with torch.no_grad():
        dyn, dl = residuals(x)
        prof = dyn.pow(2).sum(dim=(1, 2))                   # [L]
        ob = float(prof.sum().sqrt())
        depth = torch.arange(1, L + 1, device=dev, dtype=torch.float32) / L
        centroid = float((prof * depth).sum() / (prof.sum() + 1e-30))
        # source patch content pi_S over the mid band, as in the paper
        lo, hi = int(mid_band[0] * L), int(mid_band[1] * L)
        pi = max(float((C.resid[l][j] * ug_hat.squeeze()).sum())
                 for l in range(lo, hi + 1) for j in S)

    return ObstructionResult(
        ob=ob, ob_total=float((prof.sum() + dl ** 2).sqrt()),
        delivered_gap0=float(del0 / lam), delivered_gap=float(dl / lam),
        profile=prof.cpu(), depth_centroid=centroid, pi_S=pi,
        # pi floor 0.05: prevents ob_norm blowups when source content ~ 0
        # (source failures); use raw ob + pi_S separately for analysis.
        ob_norm=ob / max(pi, 0.05), cg_iters=it, cert_affine=cert)


def fragility(W: Weights, C: ForwardCache, S: Sequence[int],
              pin_L: int = None, power_iters: int = 6, cg_iters: int = 60,
              mu: float = 1e-4) -> float:
    """Affine-Laplacian spectral gap at the source interface.

    Estimates sigma_min of the pinned dynamics operator A (delivery row
    EXCLUDED) on the free coordinates, via inverse power iteration: each step
    solves (A^T A + mu) x = b with CG.  Interpretation: sigma_min is the
    minimal dynamics-defect energy per unit of state change at the interface.
    A small gap means the realized transport sits near an obstruction --
    small coefficient changes can sever delivery.  Prediction: fragile
    (small-gap) examples coincide with small decision-boundary flip radii
    and with transport-verdict instability under template paraphrase.

    Cost: power_iters * cg_iters frozen layer-stack applications.  Returns
    sigma_min estimate (mu-corrected).
    """
    L, T, d = W.L, C.resid[0].shape[0], C.resid[0].shape[1]
    dev = C.resid[0].device
    pin_L = pin_L if pin_L is not None else L // 2
    realized = torch.stack(C.resid[1:])
    mask = _free_mask(L, T, d, S, pin_L, dev)

    def dyn_res(delta):
        X = realized + delta * mask
        out, x_prev = [], C.resid[0]
        for l in range(L):
            out.append(X[l] - frozen_layer(x_prev, W, l, C))
            x_prev = X[l]
        return torch.stack(out)

    with torch.no_grad():
        dyn0 = dyn_res(torch.zeros_like(realized))

    def A_apply(delta):
        with torch.no_grad():
            return dyn_res(delta) - dyn0

    def A_adjoint(y):
        with torch.enable_grad():
            delta = torch.zeros_like(realized, requires_grad=True)
            s = (dyn_res(delta) * y).sum()
            (g,) = torch.autograd.grad(s, delta)
        return g

    def solve(b):                      # (A^T A + mu) x = b by CG
        x = torch.zeros_like(b); r = b.clone(); p = r.clone()
        rs = (r * r).sum()
        for _ in range(cg_iters):
            Ap = A_adjoint(A_apply(p)) + mu * p
            pAp = (p * Ap).sum()
            if not torch.isfinite(pAp) or pAp <= 0:
                break
            a = rs / pAp
            x, r = x + a * p, r - a * Ap
            rs_new = (r * r).sum()
            if rs_new < 1e-20:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return x

    v = torch.randn_like(realized) * mask
    v = v / v.norm()
    lam_inv = 0.0
    for _ in range(power_iters):
        w = solve(v)
        lam_inv = float(w.norm())
        v = (w / (w.norm() + 1e-30)) * mask
    lam_min = max(1.0 / max(lam_inv, 1e-30) - mu, 0.0)
    return lam_min ** 0.5
