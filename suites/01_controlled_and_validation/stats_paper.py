"""Statistics for the validation suite (paper Sec. 4.5): Wilson intervals,
Cochran-Armitage trend test, Holm correction, and logistic regression with
model fixed effects (IRLS, no statsmodels dependency)."""
import math
import torch


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def cochran_armitage(successes, totals, scores=(0, 1, 2)):
    """Two-sided trend test for proportions ordered by `scores`."""
    N = sum(totals)
    R = sum(successes)
    pbar = R / N
    sbar = sum(s * n for s, n in zip(scores, totals)) / N
    num = sum(s * k for s, k in zip(scores, successes)) - R * sbar
    var = pbar * (1 - pbar) * (
        sum(n * (s - sbar) ** 2 for s, n in zip(scores, totals)))
    if var <= 0:
        return float("nan"), float("nan")
    z = num / math.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def two_prop_p(k1, n1, k2, n2):
    """Two-sided z test for difference of proportions (pooled)."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    z = (k1 / n1 - k2 / n2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def logistic_fixed_effects(y, stage, model_id, iters=50, ridge=1e-6):
    """logit P(recover) = b0 + b1*stage + model dummies.
    Returns odds ratio per stage, Wald p, and the coefficient."""
    y = torch.tensor(y, dtype=torch.float64)
    stage = torch.tensor(stage, dtype=torch.float64)
    models = sorted(set(model_id))
    dummies = torch.stack([torch.tensor([1.0 if m == mm else 0.0
                                         for m in model_id],
                                        dtype=torch.float64)
                           for mm in models[1:]], dim=1) \
        if len(models) > 1 else torch.zeros(len(y), 0, dtype=torch.float64)
    X = torch.cat([torch.ones(len(y), 1, dtype=torch.float64),
                   stage.unsqueeze(1), dummies], dim=1)
    w = torch.zeros(X.shape[1], dtype=torch.float64)
    for _ in range(iters):
        p = torch.sigmoid(X @ w).clamp(1e-9, 1 - 1e-9)
        g = X.T @ (p - y) + ridge * w
        H = X.T @ (X * (p * (1 - p)).unsqueeze(1)) + \
            ridge * torch.eye(len(w), dtype=torch.float64)
        step = torch.linalg.solve(H, g)
        w = w - step
        if step.abs().max() < 1e-10:
            break
    p_hat = torch.sigmoid(X @ w).clamp(1e-9, 1 - 1e-9)
    H = X.T @ (X * (p_hat * (1 - p_hat)).unsqueeze(1)) + \
        ridge * torch.eye(len(w), dtype=torch.float64)
    se = torch.linalg.inv(H).diagonal().sqrt()[1].item()
    b1 = w[1].item()
    zst = b1 / se if se > 0 else float("nan")
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(zst) / math.sqrt(2))))
    return dict(odds_ratio=math.exp(b1), coef=b1, p=p, se=se)
