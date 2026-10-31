"""Where do the exact duplicates come from? (reviewer comment R1.6)

For every scenario and class this script counts exact duplicate rows in three views:
  analysis   the feature columns used in the main analysis (identifiers and clocks
             removed, as in uav_ehfs.py) plus the label: the duplicates removed
             before the split;
  raw        every column of the CSV as shipped, including addresses, ports,
             timestamps and counters: rows that are identical even here are
             repeated records, not distinct packets or flows;
  adjacent   the share of analysis-view duplicates whose copy is the immediately
             preceding row in file order (bursts of identical records).
If duplicates in the analysis view are not duplicates in the raw view, they are
distinct packets/flows that only look identical once identifiers are removed,
which is what high-rate floods produce. If they are duplicates in the raw view
too, the capture or the CSV export repeated the record.

Usage:  python review/duplicate_origin.py
Output: review/duplicate_origin/summary.json, review/duplicate_origin/per_class.csv
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import uav_ehfs as U  # noqa: E402

OUT = os.path.join(HERE, "duplicate_origin")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, summary = [], {}
    for scn in ["S1", "S2", "S3"]:
        df, label_col = U.load_raw(scn)
        y = df[label_col].map(U.canon_label).reset_index(drop=True)
        raw = df.drop(columns=[label_col]).reset_index(drop=True)

        U.KEEP_DUPLICATES = True
        X, _, _ = U.preprocess(df, label_col, "multiclass", scn)
        U.KEEP_DUPLICATES = False

        h_an = pd.util.hash_pandas_object(pd.concat([X, y.rename("__y__")], axis=1), index=False)
        h_raw = pd.util.hash_pandas_object(pd.concat([raw, y.rename("__y__")], axis=1), index=False)
        dup_an = h_an.duplicated(keep="first")
        dup_raw = h_raw.duplicated(keep="first")
        adjacent = dup_an & (h_an == h_an.shift(1))

        for c in sorted(y.unique()):
            m = y == c
            n, da, dr, adj = int(m.sum()), int(dup_an[m].sum()), int(dup_raw[m].sum()), int(adjacent[m].sum())
            rows.append({"scenario": scn, "class": c, "rows": n,
                         "dup_analysis": da, "dup_analysis_pct": 100 * da / n,
                         "dup_raw": dr, "dup_raw_pct": 100 * dr / n,
                         "share_of_analysis_dups_also_raw_dups": dr / da if da else None,
                         "share_of_analysis_dups_adjacent": adj / da if da else None})
        summary[scn] = {
            "rows": int(len(y)),
            "dup_analysis": int(dup_an.sum()),
            "dup_raw": int(dup_raw.sum()),
            "dup_analysis_pct": float(100 * dup_an.mean()),
            "dup_raw_pct": float(100 * dup_raw.mean()),
            "share_of_analysis_dups_adjacent": float(adjacent.sum() / max(dup_an.sum(), 1)),
            "n_columns_raw": int(raw.shape[1]),
            "n_columns_analysis": int(X.shape[1]),
        }
        print(scn, summary[scn], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "per_class.csv"), index=False)
    print(t.round(3).to_string())
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
