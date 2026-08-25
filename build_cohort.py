"""Build a cohort jsonl for one model.

Two sources:
  --popqa            : pull PopQA from HF (`akariasai/PopQA`), convert the
                       supported relations to completion templates, stratify
                       over subject-popularity tertiles (paper Sec. 5).
  --records f.jsonl  : your own records: {"prompt","subject","answer",
                       "aliases":[..], "donor_prompt"(opt)}.

For each record this script: generates greedily (word-level, alias-robust
match), finds the subject span under the model tokenizer, sets target first
token / competitor / target_text, builds a context-donor prompt for presence
repairs, and balances failures vs successes.

Verdicts: left null by default -- fill them from YOUR verdict-tier pipeline
(ground truth for the paper).  --standin labels with the ordered
delivery/content rule (demo-grade; fine for a first pass, not for claims).

Usage:
  python build_cohort.py --model meta-llama/Llama-3.2-3B --popqa \
      --n_per_tertile 170 --out cohorts/llama32_3b_parametric.jsonl
"""
import argparse, json, os, random, re
import torch

TEMPLATES = {   # PopQA prop -> completion template (subject placeholder)
    "capital":        "The capital of {s} is",
    "capital of":     "{s} is the capital of",
    "author":         "The author of {s} is",
    "composer":       "The composer of {s} is",
    "director":       "The director of {s} is",
    "place of birth": "{s} was born in the city of",
}


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return re.sub(r"[^\w\s]", "", s).strip()


def word_level_correct(gen, answer, aliases):
    g = norm(gen)
    return any(norm(a) and (norm(a) in g or g.startswith(norm(a)))
               for a in [answer] + list(aliases))


def find_span(prompt, substring, tok):
    enc = tok(prompt, return_offsets_mapping=True)
    a = prompt.index(substring); b = a + len(substring)
    idx = [i for i, (s, e) in enumerate(enc.offset_mapping) if s < b and e > a]
    return [min(idx), max(idx)]


@torch.no_grad()
def process(model, tok, device, prompt, subject, answer, aliases,
            donor_prompt=None, max_new=12):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    correct = word_level_correct(gen, answer, aliases)
    g = tok(" " + answer.strip(), add_special_tokens=False).input_ids[0]
    top = int(model(ids).logits[0, -1].argmax())
    span = find_span(prompt, subject, tok)
    # copy-contamination flag (paper: excluded from headline shares)
    contaminated = g in tok(prompt, add_special_tokens=False).input_ids
    donor = donor_prompt or f"{answer}. {prompt}"   # context-dominance donor
    return dict(prompt=prompt, subject=subject, target_text=" " + answer.strip(),
                target_first_token=int(g),
                competitor_token=top if top != g else -1,
                source_span=span, generated=gen, correct=bool(correct),
                verdict="correct" if correct else None,
                copy_contaminated=bool(contaminated),
                donor_prompt=donor,
                donor_source_span=find_span(donor, subject, tok),
                tau=None, ablation_effect=None)


VALUES = ("apple banana cherry mango papaya walnut copper silver cobalt "
          "amber violet crimson maple cedar willow falcon heron sparrow "
          "onyx jasper basalt tundra meadow lagoon").split()
KEYS = ("alpha bravo delta echo foxtrot kilo lima nova oscar quebec romeo "
        "sierra tango victor whiskey yankee zulu vector matrix cursor "
        "packet kernel socket buffer").split()

TWOHOP = [  # (landmark, country, capital) -- common-knowledge triples
    ("the Eiffel Tower", "France", "Paris"), ("the Colosseum", "Italy", "Rome"),
    ("the Brandenburg Gate", "Germany", "Berlin"),
    ("the Sagrada Familia", "Spain", "Madrid"),
    ("Mount Fuji", "Japan", "Tokyo"), ("the Taj Mahal", "India", "New Delhi"),
    ("the Great Wall", "China", "Beijing"),
    ("Christ the Redeemer", "Brazil", "Brasilia"),
    ("the Parthenon", "Greece", "Athens"), ("Red Square", "Russia", "Moscow"),
    ("Big Ben", "the United Kingdom", "London"),
    ("the CN Tower", "Canada", "Ottawa"),
    ("the Sydney Opera House", "Australia", "Canberra"),
    ("Machu Picchu", "Peru", "Lima"), ("the Blue Mosque", "Turkey", "Ankara"),
    ("the Pyramids of Giza", "Egypt", "Cairo"),
    ("Table Mountain", "South Africa", "Pretoria"),
    ("the Little Mermaid statue", "Denmark", "Copenhagen"),
    ("the Charles Bridge", "the Czech Republic", "Prague"),
    ("the Chain Bridge", "Hungary", "Budapest"),
    ("the Atomium", "Belgium", "Brussels"),
    ("the Rijksmuseum", "the Netherlands", "Amsterdam"),
    ("the Matterhorn", "Switzerland", "Bern"),
    ("the Hagia Sophia", "Turkey", "Ankara"),
    ("Angkor Wat", "Cambodia", "Phnom Penh"),
    ("the Petronas Towers", "Malaysia", "Kuala Lumpur"),
    ("Chichen Itza", "Mexico", "Mexico City"),
    ("the Alhambra", "Spain", "Madrid"),
    ("the Acropolis", "Greece", "Athens"),
    ("Wawel Castle", "Poland", "Warsaw"),
]


def make_extraction(n=120, n_distractors=8, seed=0):
    """Synthetic key-value extraction: the answer is verbatim in context.
    Presence holds by token identity, so all failures are transport/selection
    -- the paper's zero-presence prediction, testable with ob(s)."""
    rng = random.Random(seed)
    recs = []
    for _ in range(n):
        keys = rng.sample(KEYS, n_distractors + 1)
        vals = rng.sample(VALUES, n_distractors + 1)
        qi = rng.randrange(n_distractors + 1)
        lines = [f"{k}: {v}" for k, v in zip(keys, vals)]
        prompt = ("Registry:\n" + "\n".join(lines) +
                  f"\nThe value of {keys[qi]} is")
        recs.append(dict(prompt=prompt, subject=vals[qi],  # span = the VALUE
                         answer=vals[qi], aliases=[], task="extraction"))
    return recs


def make_twohop_popqa(n=200, seed=0):
    """Two-hop composition from the PopQA join (paper Sec. 5): entity -E->
    country (prop 'country') joined with country -C-> capital (prop
    'capital').  Sorted toward low-popularity subjects so the cohort
    actually produces failures (the curated landmark list is too easy).
    Each record carries both one-hop constituents, so 'both facts known'
    is measurable per model."""
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test")
    parse = lambda a: json.loads(a) if isinstance(a, str) else a

    capitals = {}                       # country name -> (subj, capital)
    for r in ds:
        if r["prop"] == "capital":
            ans = parse(r["possible_answers"])
            if ans:
                capitals[r["subj"].lower()] = (r["subj"], ans[0])

    recs = []
    for r in ds:
        if r["prop"] != "country":
            continue
        answers = parse(r["possible_answers"])
        hit = next((a for a in answers if a.lower() in capitals), None)
        if hit is None:
            continue
        country, cap = capitals[hit.lower()]
        recs.append(dict(
            prompt=(f"The capital of the country where {r['subj']} "
                    f"is located is"),
            subject=r["subj"], answer=cap, aliases=[], task="twohop",
            bridge=country, s_pop=r["s_pop"] or 0,
            hop1_prompt=f"{r['subj']} is located in the country of",
            hop1_answer=country,
            hop2_prompt=f"The capital of {country} is", hop2_answer=cap))
    recs.sort(key=lambda x: x["s_pop"])          # hardest (rarest) first
    rng = random.Random(seed)
    pool = recs[:2 * n]
    rng.shuffle(pool)
    return pool[:n]


def make_twohop():
    """Curated common-knowledge fallback (easy; most models near-perfect --
    prefer make_twohop_popqa for failure statistics)."""
    recs = []
    for lm, country, cap in TWOHOP:
        recs.append(dict(
            prompt=f"The capital of the country where {lm} is located is",
            subject=lm.replace("the ", "").strip(), answer=cap, aliases=[],
            task="twohop", bridge=country,
            hop1_prompt=f"{lm} is located in the country of",
            hop1_answer=country.replace("the ", ""),
            hop2_prompt=f"The capital of {country} is", hop2_answer=cap))
    return recs


def load_popqa(n_per_tertile, seed=0):
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test")
    rows = [r for r in ds if r["prop"] in TEMPLATES]
    rows.sort(key=lambda r: r["s_pop"] or 0)
    k = len(rows) // 3
    tertiles = [rows[:k], rows[k:2 * k], rows[2 * k:]]
    rng = random.Random(seed)
    picked = [r for t in tertiles for r in rng.sample(t, min(n_per_tertile, len(t)))]
    recs = []
    for r in picked:
        answers = json.loads(r["possible_answers"]) if \
            isinstance(r["possible_answers"], str) else r["possible_answers"]
        recs.append(dict(prompt=TEMPLATES[r["prop"]].format(s=r["subj"]),
                         subject=r["subj"], answer=answers[0],
                         aliases=answers[1:], relation=r["prop"],
                         s_pop=r["s_pop"] or 0))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--popqa", action="store_true")
    ap.add_argument("--task", choices=["parametric", "extraction", "twohop"],
                    default="parametric")
    ap.add_argument("--n_distractors", type=int, default=8)
    ap.add_argument("--records")
    ap.add_argument("--n_per_tertile", type=int, default=170)
    ap.add_argument("--balance", type=int, default=60,
                    help="target failures and successes per cohort (0 = keep all)")
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        attn_implementation="eager").to(args.device).eval()

    if args.task == "extraction":
        src = make_extraction(n_distractors=args.n_distractors)
    elif args.task == "twohop":
        try:
            src = make_twohop_popqa()
        except ImportError:
            print("`datasets` unavailable; using easy curated twohop list")
            src = make_twohop()
    elif args.popqa:
        src = load_popqa(args.n_per_tertile)
    else:
        src = [json.loads(l) for l in open(args.records)]
    out = []
    for i, r in enumerate(src):
        try:
            rec = process(model, tok, args.device, r["prompt"], r["subject"],
                          r["answer"], r.get("aliases", []),
                          r.get("donor_prompt"))
        except (ValueError, IndexError) as e:
            print(f"[{i}] skipped ({e})")
            continue
        for k in ("task", "bridge", "hop1_prompt", "hop1_answer",
                  "hop2_prompt", "hop2_answer", "relation", "s_pop"):
            if k in r:
                rec[k] = r[k]
        out.append(rec)
        if i % 25 == 0:
            nf = sum(not o["correct"] for o in out)
            print(f"[{i}/{len(src)}] failures so far: {nf}")

    # extraction is copy-"contaminated" BY DESIGN (target verbatim in prompt);
    # the exclusion applies only to parametric recall.
    clean = ([o for o in out if not o["copy_contaminated"]]
             if args.task == "parametric" else out)
    fails = [o for o in clean if not o["correct"]]
    succs = [o for o in clean if o["correct"]]
    if args.balance:
        rng = random.Random(0)
        fails = rng.sample(fails, min(args.balance, len(fails)))
        succs = rng.sample(succs, min(args.balance, len(succs)))
    cohort = fails + succs

    if args.standin:
        print("WARNING: stand-in verdicts are demo-grade; use your "
              "verdict-tier pipeline for paper claims.")
        from frozen_cache import Weights, build_cache, target_delivery
        W = Weights(model)
        # calibrate on successes
        caches = {}
        for o in cohort:
            ids = tok(o["prompt"], return_tensors="pt").input_ids[0].to(args.device)
            caches[id(o)] = build_cache(model, W, ids)
        dels = torch.tensor([float(target_delivery(
            caches[id(o)].resid[-1], W, caches[id(o)], o["target_first_token"]))
            for o in succs])
        ug = lambda C, t: ((C.inv_f[-1] * W.ln_f) * W.WU[t]).squeeze()
        def pi_S(C, t, S):
            lo, hi = W.L // 4, 3 * W.L // 4
            return max(float((C.resid[l][j] * ug(C, t)).sum())
                       for l in range(lo, hi + 1) for j in S)
        pis = torch.tensor([pi_S(caches[id(o)], o["target_first_token"],
                                 range(o["source_span"][0], o["source_span"][1] + 1))
                            for o in succs])
        th_d, th_p = float(dels.quantile(0.10)), float(pis.quantile(0.10))
        for o in fails:
            C = caches[id(o)]
            S = range(o["source_span"][0], o["source_span"][1] + 1)
            d = float(target_delivery(C.resid[-1], W, C, o["target_first_token"]))
            p = pi_S(C, o["target_first_token"], S)
            o["verdict"] = ("selection" if d >= th_d else
                            "transport" if p >= th_p else "presence")

    with open(args.out, "w") as f:
        for o in cohort:
            f.write(json.dumps(o) + "\n")
    from collections import Counter
    print(f"wrote {len(cohort)} records -> {args.out}")
    print("verdicts:", Counter(o["verdict"] for o in cohort))


if __name__ == "__main__":
    main()
