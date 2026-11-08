"""Statistics utilities for the cross-model claims.

homogeneity_equivalence: replaces "we do not reject homogeneity (p=0.52)"
with an affirmative equivalence bound: with 95% confidence, every pairwise
total-variation distance between model-level verdict distributions is at
most `bound`.  Small bound = models genuinely similar; a large bound honestly
reports that the cohort sizes cannot certify similarity (which is the correct
reading of a non-rejection at n~55).
"""
from collections import Counter
from itertools import combinations
import random


def _tv(labels_a, labels_b, classes):
    ca, cb = Counter(labels_a), Counter(labels_b)
    na, nb = len(labels_a), len(labels_b)
    return 0.5 * sum(abs(ca[c] / na - cb[c] / nb) for c in classes)


def homogeneity_equivalence(labels_by_model: dict, n_boot: int = 4000,
                            alpha: float = 0.05, seed: int = 0):
    """labels_by_model: {model_name: [verdict, ...]} (failures only).

    Returns (observed_max_tv, upper_bound, per_pair) where upper_bound is the
    (1-alpha) bootstrap quantile of the max pairwise TV distance.
    """
    rng = random.Random(seed)
    classes = sorted({v for ls in labels_by_model.values() for v in ls})
    models = sorted(labels_by_model)

    per_pair = {(a, b): _tv(labels_by_model[a], labels_by_model[b], classes)
                for a, b in combinations(models, 2)}
    observed = max(per_pair.values())

    maxes = []
    for _ in range(n_boot):
        res = {m: [rng.choice(labels_by_model[m])
                   for _ in labels_by_model[m]] for m in models}
        maxes.append(max(_tv(res[a], res[b], classes)
                         for a, b in combinations(models, 2)))
    maxes.sort()
    bound = maxes[min(int((1 - alpha) * n_boot), n_boot - 1)]
    return observed, bound, per_pair


if __name__ == "__main__":
    import sys, csv
    from collections import defaultdict
    # usage: python stats_utils.py results/*/obstruction.csv
    labels = defaultdict(list)
    for path in sys.argv[1:]:
        model = path.split("/")[-2]
        for row in csv.DictReader(open(path)):
            if row["verdict"] != "correct":
                labels[model].append(row["verdict"])
    obs, bound, pairs = homogeneity_equivalence(dict(labels))
    print(f"models: {len(labels)}; observed max pairwise TV = {obs:.3f}")
    print(f"95% equivalence bound: all pairwise TV <= {bound:.3f}")
    worst = max(pairs, key=pairs.get)
    print(f"least similar pair: {worst[0]} vs {worst[1]} "
          f"(TV = {pairs[worst]:.3f})")
