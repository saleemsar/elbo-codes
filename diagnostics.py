"""Duplicate-structure diagnostics for S3 (used in the sensitivity paragraph).
Reads the with-duplicates S3 multi-class predictions and writes results/diagnostics.json."""
import json
import os

import pandas as pd

import uav_ehfs as U

U.KEEP_DUPLICATES = True
df, label = U.load_raw("S3")
X, y, _ = U.preprocess(df, label, "multiclass")
h = pd.util.hash_pandas_object(X, index=False).values
labels_per_vector = pd.Series(y.values).groupby(h).agg(lambda s: set(s))
conflicting = labels_per_vector[labels_per_vector.map(len) > 1]
p = pd.read_csv(os.path.join(U.OUT_DIR, "with_duplicates", "S3_multiclass_test_predictions.csv.gz"))
p["vector"] = h[p.row_id.values]
mis = p[(p.y_true == "DoS") & (p.y_pred_task_aware != "DoS")]
out = {"S3_conflicting_vectors": int(len(conflicting)),
       "S3_conflicting_vectors_involving_DoS": int(sum("DoS" in s for s in conflicting)),
       "S3_distinct_DoS_vectors": int(len(set(h[(y == "DoS").values]))),
       "S3_withdup_misclassified_DoS_test_rows": int(len(mis)),
       "S3_withdup_misclassified_DoS_distinct_vectors": int(mis.vector.nunique()),
       "S3_withdup_misclassified_DoS_largest_vector_count": int(mis.vector.value_counts().iloc[0])}
with open(os.path.join(U.OUT_DIR, "diagnostics.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(out)
