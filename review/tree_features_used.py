"""Number of input features the final trees actually split on (reviewer comment R1.10).

A tree only needs the fields it splits on at inference time, which can be fewer than
the subset it was trained on. For the task-aware subset and for all features, with
the unconstrained and the size-constrained tree (review/baselines.py, part pruning),
the trees are refitted exactly as in review/deployment_cost.py and the distinct
split features are counted.

Usage:  python review/tree_features_used.py
Output: review/deployment_cost/features_used.json
"""
import json
import os
import sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import uav_ehfs as U  # noqa: E402

RUNS = [(s, t) for s in ["S1", "S2", "S3"] for t in ["binary", "multiclass"]]


def used(d, F, model):
    A, _, _ = U.fit_transform_fold(d["Xtr"][:, F], d["Xte"][:, F])
    As, ys = U.safe_smote(A, d["ytr"])
    model.fit(As, ys)
    f = model.tree_.feature
    return sorted({d["feat_names"][F[j]] for j in f[f >= 0]})


def main():
    out = {}
    for scn, task in RUNS:
        res = json.load(open(os.path.join(U.OUT_DIR, f"{scn}_{task}.json")))
        prune = json.load(open(os.path.join(HERE, "baselines", f"{scn}_{task}_pruning.json")))
        d = U.prepare(scn, task)
        F_star = sorted(x["index"] for x in res["selection"]["selected_features"])
        entry = {}
        for tag, F in [("task_aware", F_star), ("all_features", list(range(len(d["feat_names"]))))]:
            c = prune[tag]["chosen"]
            for kind, m in [("unconstrained", U.detector()),
                            ("constrained", DecisionTreeClassifier(criterion="gini", class_weight="balanced",
                                                                   random_state=U.SEED, max_depth=c["max_depth"],
                                                                   min_samples_leaf=c["min_samples_leaf"]))]:
                u = used(d, F, m)
                entry[f"{tag}|{kind}"] = {"n_subset": len(F), "n_used": len(u), "used": u}
        out[f"{scn}_{task}"] = entry
        print(scn, task, {k: (v["n_subset"], v["n_used"]) for k, v in entry.items()}, flush=True)
    with open(os.path.join(HERE, "deployment_cost", "features_used.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
