"""Explicit paper input-provenance and legacy edge-sum measurements."""
import torch
from validation_suite import make_api


def supported(model):
    kind = model.config.model_type
    if getattr(model.get_output_embeddings(), 'bias', None) is not None:
        raise ValueError('Biased output head not supported by readout repair adapter')
    if kind not in ('llama', 'qwen2', 'gpt_neox'):
        raise ValueError(f'Unsupported architecture {kind}; do not bypass reconstruction checks')
    if getattr(model.config, 'sliding_window', None) and getattr(model.config, 'use_sliding_window', False):
        raise ValueError('Sliding-window attention is not supported')
    if kind != 'gpt_neox':
        for b in model.model.layers:
            mods = [b.self_attn.o_proj, b.mlp.up_proj, b.mlp.gate_proj,
                    b.mlp.down_proj, b.input_layernorm, b.post_attention_layernorm]
            if any(getattr(m, 'bias', None) is not None for m in mods):
                raise ValueError('Original frozen adapter omits these bias terms')
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    return make_api(model)


def layer_fn(W):
    if W.__class__.__name__ == 'NeoXWeights':
        from frozen_neox import frozen_layer_neox
        return frozen_layer_neox
    from frozen_cache import frozen_layer
    return frozen_layer


def linear_layer(x, W, l, C):
    f = layer_fn(W)
    # Subtract affine offset exactly: do not inject biases into each labelled source.
    return f(x, W, l, C) - f(torch.zeros_like(x), W, l, C)


def read_component(W, C, x, token):
    if W.__class__.__name__ == 'NeoXWeights':
        from frozen_neox import ln_apply
        # Final LayerNorm is affine under cached scales; remove its bias.
        return float((ln_apply(x[-1:], C.inv_f[-1:], W.lnfw, W.lnfb)[0]
                      - ln_apply(torch.zeros_like(x[-1:]), C.inv_f[-1:], W.lnfw, W.lnfb)[0]) @ W.WU[token])
    from frozen_cache import target_delivery
    return float(target_delivery(x, W, C, token))


@torch.no_grad()
def provenance(W, C, S, g, cut=0):
    x = torch.zeros_like(C.resid[cut]); x[S] = C.resid[cut][S]
    for l in range(cut, W.L): x = linear_layer(x, W, l, C)
    return read_component(W, C, x, g)


@torch.no_grad()
def partition_certificate(W, C, S, tol=1e-4):
    """Source + complement + accumulated bias reconstruct every cut."""
    f = layer_fn(W)
    source = torch.zeros_like(C.resid[0]); source[S] = C.resid[0][S]
    rest = C.resid[0] - source; bias = torch.zeros_like(source)
    worst = 0.
    for l in range(W.L):
        source = linear_layer(source, W, l, C)
        rest = linear_layer(rest, W, l, C)
        bias = f(bias, W, l, C)
        err = float((source + rest + bias - C.resid[l+1]).norm() /
                    C.resid[l+1].norm().clamp_min(1e-12))
        worst = max(worst, err)
    if worst >= tol: raise AssertionError(f'Provenance partition failed: {worst}')
    return worst


@torch.no_grad()
def lens_matrix(api, C, S, tokens):
    return torch.stack([torch.stack([api.lens(C, l, i, tokens)
                        for l in range(1, api.L+1)]) for i in S]).cpu().numpy()


@torch.no_grad()
def readout_repairs(api, C, g, c, med_g, med_c, doses, seed=0):
    """Monotone calibrated edits and norm-matched controls; normalized-state edits."""
    x = api.final_state(C); ug = api.WU[g]; uc = api.WU[c]
    eg = ug / ug.norm(); ec = uc / uc.norm()
    updose = max(0., med_g-float(x @ eg)); downdose = max(0., float(x @ ec)-med_c)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    rand = torch.randn(x.shape, generator=rng).to(x); rand /= rand.norm()
    edits = [('answer_up', x+updose*eg), ('alt_down', x-downdose*ec),
             ('combined', x+updose*eg-downdose*ec), ('random', x+updose*rand)]
    for dose in doses:
        edits.extend([(f'fixed_answer_{dose:g}', x+dose*eg),
                      (f'fixed_random_{dose:g}', x+dose*rand)])
    rows=[]
    baseline = x @ api.WU.T
    for name, new in edits:
        dz = (new-x) @ api.WU.T
        z = baseline+dz
        rows.append(dict(arm=name, recovered=int(z.argmax())==g,
                         dose_norm=float((new-x).norm()),
                         delta_gold=float(dz[g]), delta_margin=float(dz[g]-dz[c]),
                         new_margin=float(z[g]-z[c])))
    return rows
