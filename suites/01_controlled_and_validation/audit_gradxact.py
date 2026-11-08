"""Baseline audit: establish ranker sign conventions on an analytic model
BEFORE evaluating real models.  (The observed rho ~ -0.79 for gradxact was
suspiciously systematic; do not flip signs post hoc -- derive them here.)

Analytic model: margin m(s_1..s_H) = u . sum_h W_h s_h, purely linear in the
per-head source contributions s_h.  Ablating head h's source part exactly
changes the margin by

    drop_h = m0 - m(without s_h) = u . W_h s_h          (exact, no approx)

First-order predictor from gradient x activation:
    grad_h = dm/ds_h = W_h^T u ;   predictor = <grad_h, s_h> = u . W_h s_h

=> the CORRECT convention is  +<grad, source part>,  which equals the exact
drop on the linear model.  DLA (u . W_h s_h) is likewise exactly the drop.
Any implementation must reproduce drop_h with rho = +1.0 here; a systematic
-1.0 indicates a sign error in the implementation, not in the mathematics.

Run:  python audit_gradxact.py    (CPU, <1 s; asserts, then prints verdict)
"""
import torch


def analytic_audit(H=12, dh=16, d=32, seed=0, tol=1e-5):
    g = torch.Generator().manual_seed(seed)
    W_h = torch.randn(H, d, dh, generator=g)
    s = torch.randn(H, dh, generator=g)
    u = torch.randn(d, generator=g)

    m0 = sum((u @ (W_h[h] @ s[h])) for h in range(H))
    exact_drop = torch.tensor([float(u @ (W_h[h] @ s[h]))
                               for h in range(H)])

    # gradient x activation, computed by autograd exactly as ranking.py does
    s_leaf = s.clone().requires_grad_(True)
    m = sum((u @ (W_h[h] @ s_leaf[h])) for h in range(H))
    m.backward()
    gxa_plus = (s_leaf.grad * s).sum(-1)          # +<grad, source part>
    gxa_minus = -gxa_plus

    # DLA on the same model
    dla = torch.tensor([float(u @ (W_h[h] @ s[h])) for h in range(H)])

    from scipy.stats import spearmanr
    rho_plus = spearmanr(gxa_plus, exact_drop).statistic
    rho_minus = spearmanr(gxa_minus, exact_drop).statistic
    err = (gxa_plus - exact_drop).abs().max()

    assert err < tol, f"gxa_plus must EQUAL exact drop on linear model ({err})"
    assert rho_plus > 0.999 and rho_minus < -0.999
    assert spearmanr(dla, exact_drop).statistic > 0.999
    return dict(rho_plus=float(rho_plus), rho_minus=float(rho_minus),
                max_err=float(err))


if __name__ == "__main__":
    r = analytic_audit()
    print("analytic audit (linear model, exact ablation effects):")
    print(f"  +<grad, source part>  rho = {r['rho_plus']:+.3f}  "
          f"(max |pred - exact| = {r['max_err']:.1e})")
    print(f"  -<grad, source part>  rho = {r['rho_minus']:+.3f}")
    print("VERDICT: the correct gradxact convention is +<grad, source part>."
          "\nranking.py must use the + sign; frozen before real-model eval.")
