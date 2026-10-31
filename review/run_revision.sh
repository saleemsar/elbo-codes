#!/usr/bin/env bash
# Every computation added in revision 1 (PONE-D-26-48338), in the order it was run.
# Main results (clean feature set, ANOVA F Stage 2, median imputation) go to results/;
# sensitivity analyses to results/sensitivity/<name>/; stability repeats to
# results/stability/seed_<s>/; label-coding repeats to results/label_coding/perm_<p>/.
# First-round results are kept in results/round1/.
#
#   bash review/run_revision.sh main|sensitivity|stability|labels|review|all
set -uo pipefail   # no -e: one failing configuration must not stop the others
cd "$(dirname "$0")/.."
PY=${PY:-python}
export UAV_EHFS_JOBS=${UAV_EHFS_JOBS:-5}

main() {
  $PY review/feature_audit.py                 # A1: feature validity audit -> review/feature_audit/
  $PY uav_ehfs.py all                         # main analysis -> results/
}

sensitivity() {
  $PY uav_ehfs.py all --fast --features validated --out results/sensitivity/validated
  $PY uav_ehfs.py all --fast --stage2 mi         --out results/sensitivity/stage2_mi
  $PY uav_ehfs.py all --fast --stage2 pearson    --out results/sensitivity/stage2_pearson
  $PY uav_ehfs.py all --fast --impute mean       --out results/sensitivity/impute_mean
  $PY uav_ehfs.py all --fast --impute zero       --out results/sensitivity/impute_zero
  $PY uav_ehfs.py all --fast --keep-duplicates   --out results/sensitivity/with_duplicates
  # first-round settings, to check that this machine reproduces results/round1/
  $PY uav_ehfs.py all --fast --features published --stage2 pearson --impute mean --out results/sensitivity/round1_rerun
}

stability() {
  for s in 1 2 3 4 5 6 7 8 9 10; do
    $PY uav_ehfs.py all --fast --seed "$s" --out "results/stability/seed_$s"
  done
}

labels() {   # Pearson Stage 2 under random class codings (multi-class only)
  for p in 1 2 3 4 5; do
    for s in S1 S2 S3; do
      [ -f "results/label_coding/perm_$p/${s}_multiclass.json" ] && continue   # already computed
      $PY uav_ehfs.py "$s" multiclass --fast --stage2 pearson --label-perm "$p" --out "results/label_coding/perm_$p"
    done
  done
}

review() {
  $PY review/duplicate_origin.py              # R1.6
  $PY review/smote_placement.py all           # R2.4 (E1/E2 on the revised pipeline)
  $PY review/baselines.py all                 # R1.2, R1.9, R1.11, R2.3
  $PY review/evaluation_checks.py             # R1.3, R1.4, R1.12, R2.6
  $PY review/deployment_cost.py               # R1.5, R1.10
  $PY review/tree_features_used.py            # R1.10 (features read at inference)
}

one() {   # bash review/run_revision.sh one S2 multiclass [extra uav_ehfs options]
  $PY uav_ehfs.py "$@"
}

case "${1:-all}" in
  one) shift; one "$@" ;;
  main) main ;;
  sensitivity) sensitivity ;;
  stability) stability ;;
  labels) labels ;;
  review) review ;;
  all) main; sensitivity; stability; labels; review ;;
  *) echo "unknown step $1"; exit 1 ;;
esac
