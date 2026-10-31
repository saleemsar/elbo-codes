"""Revision-1 tables and in-text numbers (feature audit, sensitivity, stability,
label coding, other detectors and selectors, constrained trees, deployment cost,
evaluation checks, duplicate origin). Executed by make_tables.py, which provides
R, R1, M, SCN, TASKS and the table helpers; every block is skipped when its input
file has not been produced yet.
"""
import csv
import json
import os

import numpy as np


def build(g):
    R, R1, M, SCN, TASKS = g["R"], g["R1"], g["M"], g["SCN"], g["TASKS"]
    SCN_LABEL, MACRO_SCN, MACRO_TASK = g["SCN_LABEL"], g["MACRO_SCN"], g["MACRO_TASK"]
    f4, pct, i, tex, write, tabular, load = g["f4"], g["pct"], g["i"], g["tex"], g["write"], g["tabular"], g["load"]
    HERE, RES = g["HERE"], g["RES"]
    REV = os.path.join(HERE, "review")

    def jload(*parts):
        path = os.path.join(*parts)
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return json.load(fh)

    def safe(text):
        """Escape free text (raw CSV values) for LaTeX."""
        rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "%": r"\%",
               "#": r"\#", "_": r"\_", "^": r"\^{}", "~": r"\~{}", "|": "/", "<": r"$<$", ">": r"$>$"}
        return "".join(rep.get(ch, ch) for ch in str(text))

    def jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if a | b else 1.0

    def feats(r):
        return [d["feature"] for d in r["selection"]["selected_features"]]

    def tm(r, tag="task_aware"):
        m = r["test"][tag]
        return m["weighted"]["f1"], m["macro"]["f1"]

    def key_macro(key):
        """Option name -> macro fragment; LaTeX macro names cannot contain digits."""
        words = {"0": "zero", "1": "one", "2": "two", "3": "three"}
        return "".join(w.capitalize() for w in "".join(words.get(ch, ch) for ch in key).split("_"))

    FSTAR, ALL = r"$\mathcal{F}^{*}$", "All"

    # ------------------------------------------------------------ feature audit (R1.7, R2.5)
    AUD = jload(REV, "feature_audit", "summary.json")
    if AUD:
        rows = []
        for s in SCN:
            with open(os.path.join(REV, "feature_audit", f"{s}_columns.csv")) as fh:
                cols = list(csv.DictReader(fh))
            r = R[(s, "multiclass")]
            n_id = sum(c["status"] == "identifier" for c in cols)
            n_inv = sum(c["status"] == "invalid" for c in cols)
            n_val = sum(c["status"] == "validated" for c in cols)
            # identifiers found by content only (header names an unrelated field)
            clocks = sum(c["status"] == "identifier" and "header" not in c["reason"] for c in cols)
            rows.append([SCN_LABEL[s], len(cols), n_id, clocks, n_inv, n_val,
                         r["preprocessing"]["n_features_after_preprocessing"]])
            for k, v in [("AudIdent", n_id), ("AudInvalid", n_inv), ("AudValid", n_val), ("AudClock", clocks),
                         ("AudRaw", len(cols))]:
                M[MACRO_SCN[s] + k] = str(v)
        write("tab_audit.tex", tabular("|l|r|r|r|r|r|r|",
              ["Scenario", "Raw columns", "Identifier", "by content", "Unverified", "Validated", "Used ($n$)"], rows))
        # Full column list (S4 Table), as a longtable
        lines = [r"\begin{longtable}{|l|p{1.9in}|l|r|p{3.2in}|}", r"\hline",
                 r"\textbf{Scenario} & \textbf{Column} & \textbf{Status} & \textbf{In domain (\%)} & \textbf{Reason / content}\\ \thickhline",
                 r"\endhead"]
        for s in SCN:
            with open(os.path.join(REV, "feature_audit", f"{s}_columns.csv")) as fh:
                for c in csv.DictReader(fh):
                    dom = c.get("in_domain_share") or ""
                    dom = f"{100 * float(dom):.1f}" if dom not in ("", "nan") else "---"
                    why = c["reason"] if c["reason"] else "values match header domain"
                    ex = c.get("examples", "")
                    if c["status"] != "validated" and ex:
                        why += "; e.g. " + ex[:60]
                    lines.append(" & ".join([s, r"\texttt{" + tex(c["column"]) + "}", c["status"], dom,
                                             safe(why)]) + r"\\ \hline")
        lines.append(r"\end{longtable}")
        write("tab_audit_columns.tex", "\n".join(lines) + "\n")
        al = AUD.get("S3_alignment")
        if al:
            rows = [[tex(a["identity"]), pct(a["as_shipped"], 1), pct(a["realigned"], 1), i(a["n_rows"])] for a in al]
            write("tab_s3_alignment.tex", tabular("|l|r|r|r|",
                  ["Identity", r"Header as shipped (\%)", r"Realigned (\%)", "Informative rows"], rows))
            M["AlignMinPct"] = pct(min(a["realigned"] for a in al), 1)
            M["AlignMaxShippedPct"] = pct(max(a["as_shipped"] for a in al), 1)

    # ------------------------------------------------------------ elbow SE values (R1.15) and K_hat (R2.3)
    rows = []
    for s in SCN:
        for t in TASKS:
            r = R[(s, t)]
            sel = r["selection"]
            K = sel["K"]
            ih, ik = K.index(sel["K_hat_argmax"]), K.index(sel["K_star"])
            kh = r["test"].get("k_hat")
            rows.append([SCN_LABEL[s], t.capitalize(), r["preprocessing"]["n_features_after_preprocessing"],
                         sel["K_star"], sel["K_hat_argmax"], sel["K_wcss_kneedle"] or "---",
                         f4(sel["cv_f1_mean"][ih]), f4(sel["cv_f1_se"][ih]), f4(sel["one_se_threshold"]),
                         f4(sel["cv_f1_mean"][ik]), f"{sel['reduction_pct']:.1f}"])
            p = MACRO_SCN[s] + MACRO_TASK[t]
            M[p + "Nsel"] = str(sel["n_selected"])          # |F*| (can be < K* when K-means leaves clusters empty)
            M[p + "NselHat"] = str(len(sel["k_hat_features"]))
            M[p + "SEhat"] = f4(sel["cv_f1_se"][ih])
            M[p + "CVhat"] = f4(sel["cv_f1_mean"][ih])
            M[p + "Thr"] = f4(sel["one_se_threshold"])
            if kh:
                M[p + "KhatwF"] = pct(kh["weighted"]["f1"])
                M[p + "KhatmF"] = pct(kh["macro"]["f1"])
    write("tab_elbow.tex", tabular("|l|l|r|r|r|r|r|r|r|r|r|",
          ["Scenario", "Task", "$n$", "$K^{*}$", r"$\hat{K}$", r"$K_{\mathrm{WCSS}}$", r"$\bar{F}_1(\hat{K})$",
           r"SE$(\hat{K})$", "Threshold", r"$\bar{F}_1(K^{*})$", r"Red.\ (\%)"], rows))

    # ------------------------------------------------------------ per-class with Wilson intervals (R1.4)
    EVC = {k: jload(REV, "evaluation_checks", f"{k}.json")
           for k in ["minority", "overlap", "blocked", "confusion", "smote_s1"]}
    if EVC["minority"]:
        for s in SCN:
            mi = EVC["minority"][f"{s}_multiclass"]
            m = R[(s, "multiclass")]["test"]["task_aware"]
            rows = []
            for c, pc in zip(mi["per_class"], m["per_class"]):
                lo, hi = c["recall_ci95"]
                flag = "" if c["estimable"] else r"$^{\dagger}$"
                rows.append([c["class"] + flag, f4(pc["precision"]), f4(pc["recall"]),
                             f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "---", f4(pc["f1"]), i(pc["support"])])
            rows.append(["Macro avg", f4(m["macro"]["precision"]), f4(m["macro"]["recall"]), "",
                         f4(m["macro"]["f1"]), i(m["n_test"])])
            rows.append([r"Macro avg, support $\geq 30$", "", "", "", f4(mi["macro_f1_estimable_only"]), ""])
            rows.append(["Weighted avg", f4(m["weighted"]["precision"]), f4(m["weighted"]["recall"]), "",
                         f4(m["weighted"]["f1"]), i(m["n_test"])])
            write(f"tab_perclass_{s}.tex", tabular("|l|r|r|l|r|r|",
                  ["Class", "Precision", "Recall", r"Recall 95\% CI", "F1", "Support"], rows))
            M[MACRO_SCN[s] + "MultimFEst"] = pct(mi["macro_f1_estimable_only"])
            M[MACRO_SCN[s] + "MultiNotEst"] = str(mi["n_not_estimable"])
        M["SThreeBinmFEst"] = pct(EVC["minority"]["S3_binary"]["macro_f1_estimable_only"])

    # ------------------------------------------------------------ sensitivity (feature set, Stage 2, imputation, duplicates)
    SENS = [("main", "Main (clean set, ANOVA F, median)", R), ("round1", "Identifiers by name and address only (clocks kept), Pearson, mean", R1)]
    for key, lab in [("validated", "Header-validated features only"), ("stage2_mi", "Stage 2: mutual information"),
                     ("stage2_pearson", "Stage 2: Pearson (integer-coded labels)"), ("impute_mean", "Mean imputation"),
                     ("impute_zero", "Zero imputation"), ("with_duplicates", "Duplicates kept")]:
        SENS.append((key, lab, load(os.path.join("sensitivity", key))))
    rows = []
    for key, lab, D in SENS:
        if any((s, t) not in D for s in SCN for t in TASKS):
            continue
        row = [lab]
        for s in SCN:
            for t in TASKS:
                r = D[(s, t)]
                w, mf = tm(r)
                row += [f"{r['selection']['K_star']}/{r['preprocessing']['n_features_after_preprocessing']}",
                        f"{w:.3f}/{mf:.3f}"]
                p = MACRO_SCN[s] + MACRO_TASK[t] + key_macro(key)
                M[p + "wF"], M[p + "mF"] = pct(w), pct(mf)
                M[p + "Kstar"] = str(r["selection"]["K_star"])
                M[p + "n"] = str(r["preprocessing"]["n_features_after_preprocessing"])
        rows.append(row)
    if rows:
        hdr = ["Configuration"] + [x for s in SCN for t in TASKS
                                   for x in [f"{s} {'Bin' if t == 'binary' else 'MC'} $K^{{*}}$/$n$", "wF1/mF1"]]
        write("tab_sensitivity.tex", tabular("|l|" + "r|" * 12, hdr, rows))
    for key in ["stage2_mi", "stage2_pearson", "impute_mean", "impute_zero", "validated"]:
        D = load(os.path.join("sensitivity", key))
        for s in SCN:
            for t in TASKS:
                if (s, t) in D:
                    M[MACRO_SCN[s] + MACRO_TASK[t] + key_macro(key) + "Jacc"] = \
                        f"{jaccard(feats(D[(s, t)]), feats(R[(s, t)])):.2f}"
    D = load(os.path.join("sensitivity", "stage2_pearson"))
    if all((s, "binary") in D for s in SCN):
        M["NBinPearsonSame"] = str(sum(sorted(feats(D[(s, "binary")])) == sorted(feats(R[(s, "binary")])) for s in SCN))

    # ------------------------------------------------------------ Pearson under random class codings (R2.1)
    LC = os.path.join(RES, "label_coding")
    ref = load(os.path.join("sensitivity", "stage2_pearson"))
    if os.path.isdir(LC):
        rows = []
        for s in SCN:
            runs = [r for r in (jload(LC, d, f"{s}_multiclass.json") for d in sorted(os.listdir(LC))) if r]
            if not runs or (s, "multiclass") not in ref:
                continue
            base = ref[(s, "multiclass")]
            ks = [r["selection"]["K_star"] for r in runs]
            jac = [jaccard(feats(r), feats(base)) for r in runs]
            mf = [tm(r)[1] for r in runs]
            rows.append([SCN_LABEL[s], len(runs), base["selection"]["K_star"], f"{min(ks)}--{max(ks)}",
                         sum(j < 1 for j in jac), f"{min(jac):.2f}", f"{min(mf):.3f}--{max(mf):.3f}"])
            p = MACRO_SCN[s]
            M[p + "LcKmin"], M[p + "LcKmax"] = str(min(ks)), str(max(ks))
            M[p + "LcJmin"], M[p + "LcNdiff"] = f"{min(jac):.2f}", str(sum(j < 1 for j in jac))
            M[p + "LcmFmin"], M[p + "LcmFmax"] = pct(min(mf)), pct(max(mf))
            M["LcNperm"] = str(len(runs))
        if rows:
            write("tab_label_coding.tex", tabular("|l|r|r|r|r|r|r|",
                  ["Scenario", "Codings", "$K^{*}$, alphabetical", "$K^{*}$ range", "Different subsets",
                   "Min.\\ Jaccard", "Test mF1 range"], rows))

    # ------------------------------------------------------------ resampling stability (R2.2, R2.4)
    ST = os.path.join(RES, "stability")
    if os.path.isdir(ST):
        rows = []
        for s in SCN:
            for t in TASKS:
                runs = [R[(s, t)]] + [r for r in (jload(ST, d, f"{s}_{t}.json") for d in sorted(os.listdir(ST))) if r]
                if len(runs) < 3:
                    continue
                ks = np.array([r["selection"]["K_star"] for r in runs])
                sets = [feats(r) for r in runs]
                jac = [jaccard(a, b) for k, a in enumerate(sets) for b in sets[k + 1:]]
                w = np.array([tm(r)[0] for r in runs])
                mf = np.array([tm(r)[1] for r in runs])
                freq = {}
                for st in sets:
                    for f in st:
                        freq[f] = freq.get(f, 0) + 1
                core = sum(v == len(runs) for v in freq.values())
                rows.append([SCN_LABEL[s], t.capitalize(), len(runs), f"{int(np.median(ks))} [{ks.min()}--{ks.max()}]",
                             f"{np.mean(jac):.2f}", core,
                             f"{w.mean():.4f} [{np.quantile(w, .025):.4f}, {np.quantile(w, .975):.4f}]",
                             f"{mf.mean():.4f} [{np.quantile(mf, .025):.4f}, {np.quantile(mf, .975):.4f}]"])
                p = MACRO_SCN[s] + MACRO_TASK[t]
                M[p + "StKmin"], M[p + "StKmax"], M[p + "StKmed"] = str(ks.min()), str(ks.max()), str(int(np.median(ks)))
                M[p + "StJacc"], M[p + "StCore"] = f"{np.mean(jac):.2f}", str(core)
                M[p + "StmFlo"], M[p + "StmFhi"] = pct(np.quantile(mf, .025)), pct(np.quantile(mf, .975))
                M[p + "StwFlo"], M[p + "StwFhi"] = pct(np.quantile(w, .025)), pct(np.quantile(w, .975))
                M["StNruns"] = str(len(runs) - 1)       # stability seeds (excluding the main run)
                M["StNtotal"] = str(len(runs))          # main run + seeds
        if rows:
            write("tab_stability.tex", tabular("|l|l|r|r|r|r|l|l|",
                  ["Scenario", "Task", "Runs", "$K^{*}$ median [range]", "Mean Jaccard", "Always selected",
                   r"Test wF1 mean [2.5\%, 97.5\%]", r"Test mF1 mean [2.5\%, 97.5\%]"], rows))

    # ------------------------------------------------------------ other detectors (R1.2)
    DET = {(s, t): jload(REV, "baselines", f"{s}_{t}_detectors.json") for s in SCN for t in TASKS}
    DN = [("cart", "CART"), ("logistic_regression", "Logistic regression"), ("linear_svm", "Linear SVM"),
          ("random_forest", "Random forest")]
    if all(DET.values()):
        for t in TASKS:
            rows = []
            for dk, dl in DN:
                row = [dl]
                for s in SCN:
                    for sub in ["task_aware", "all_features"]:
                        x = DET[(s, t)][f"{sub}|{dk}"]
                        row.append(f"{x['weighted_f1']:.3f}/{x['macro_f1']:.3f}")
                rows.append(row)
            write(f"tab_detectors_{t}.tex", tabular("|l|r|r|r|r|r|r|",
                  ["Detector"] + [f"{s}, {lab}" for s in SCN for lab in [FSTAR, ALL]], rows))
        for s in SCN:
            for t in TASKS:
                for dk, _ in DN:
                    e = DET[(s, t)]
                    d = e[f"task_aware|{dk}"]["macro_f1"] - e[f"all_features|{dk}"]["macro_f1"]
                    M[MACRO_SCN[s] + MACRO_TASK[t] + "Det" + key_macro(dk) + "DmF"] = \
                        f"{100 * d:+.1f}".replace("-", "$-$")
                    M[MACRO_SCN[s] + MACRO_TASK[t] + "Det" + key_macro(dk) + "mF"] = pct(e[f"task_aware|{dk}"]["macro_f1"])

    # ------------------------------------------------------------ other selectors (R1.9, R2.3)
    SN = [("variance", "Variance"), ("pca", "PCA (components)"), ("anova_f", "ANOVA F top-$k$"),
          ("mutual_info", "Mutual information top-$k$"), ("chi2", r"$\chi^2$ top-$k$"), ("mrmr", "mRMR"),
          ("forest_importance", "Forest importance"), ("l1_logistic", "L1 logistic"), ("rfe_cart", "RFE (CART)")]
    SEL = {(s, t): jload(REV, "baselines", f"{s}_{t}_selectors.json") for s in SCN for t in TASKS}
    if all(SEL.values()):
        for t in TASKS:
            row = ["UAV-EHFS (task-aware elbow)"]
            for s in SCN:
                r = R[(s, t)]
                w, mf = tm(r)
                row += [r["selection"]["K_star"], f"{w:.3f}/{mf:.3f}", "---"]
            rows = [row]
            for sk, sl in SN:
                row = [sl]
                for s in SCN:
                    e = SEL[(s, t)][sk]
                    row += [e["k_own"], f"{e['own_k']['weighted_f1']:.3f}/{e['own_k']['macro_f1']:.3f}",
                            f"{e['matched_K_star']['weighted_f1']:.3f}/{e['matched_K_star']['macro_f1']:.3f}"]
                rows.append(row)
            write(f"tab_selectors_{t}.tex", tabular("|l|" + "r|r|r|" * 3,
                  ["Selector"] + [x for s in SCN for x in [f"{s} $k^{{*}}$", "own $k^{*}$", "$k = K^{*}$"]], rows))
            for s in SCN:
                own = [SEL[(s, t)][k]["own_k"]["macro_f1"] for k, _ in SN]
                p = MACRO_SCN[s] + MACRO_TASK[t]
                M[p + "SelMinmF"], M[p + "SelMaxmF"] = pct(min(own)), pct(max(own))
                M[p + "SelNkOne"] = str(sum(SEL[(s, t)][k]["k_own"] == 1 for k, _ in SN))
                M[p + "SelNabove"] = str(sum(v > R[(s, t)]["test"]["task_aware"]["macro"]["f1"] + 1e-9 for v in own))
                best = max(SN, key=lambda kv: SEL[(s, t)][kv[0]]["own_k"]["macro_f1"])
                M[MACRO_SCN[s] + MACRO_TASK[t] + "BestSel"] = best[1]
                M[MACRO_SCN[s] + MACRO_TASK[t] + "BestSelmF"] = pct(SEL[(s, t)][best[0]]["own_k"]["macro_f1"])
                M[MACRO_SCN[s] + MACRO_TASK[t] + "BestSelk"] = str(SEL[(s, t)][best[0]]["k_own"])

    # ------------------------------------------------------------ constrained trees (R1.11) and cost (R1.5, R1.10)
    PR = {(s, t): jload(REV, "baselines", f"{s}_{t}_pruning.json") for s in SCN for t in TASKS}
    DC = jload(REV, "deployment_cost", "summary.json")
    FU = jload(REV, "deployment_cost", "features_used.json")
    if FU:
        for s in SCN:
            for t in TASKS:
                for sub in ["task_aware", "all_features"]:
                    for kind in ["unconstrained", "constrained"]:
                        p = MACRO_SCN[s] + MACRO_TASK[t] + ("Ta" if sub == "task_aware" else "All") + \
                            ("Un" if kind == "unconstrained" else "Pr") + "Used"
                        M[p] = str(FU[f"{s}_{t}"][f"{sub}|{kind}"]["n_used"])
    if all(PR.values()):
        rows = []
        for s in SCN:
            for t in TASKS:
                for sub, lab in [("task_aware", FSTAR), ("all_features", ALL)]:
                    e = PR[(s, t)][sub]
                    u, c, ch = e["unconstrained"], e["constrained"], e["chosen"]
                    row = [SCN_LABEL[s], "Bin" if t == "binary" else "MC", lab, u["n_features"],
                           f"{u['weighted_f1']:.3f}/{u['macro_f1']:.3f}", i(u["tree_nodes"]), f"{u['c_array_kb']:.0f}",
                           f"{ch['max_depth'] if ch['max_depth'] else 'none'}/{ch['min_samples_leaf']}",
                           f"{c['weighted_f1']:.3f}/{c['macro_f1']:.3f}", i(c["tree_nodes"]), f"{c['c_array_kb']:.1f}"]
                    if DC:
                        dc = DC[f"{s}_{t}"].get(f"{sub}|constrained")
                        row.append(f"{dc['mean_path_length']:.1f}" if dc else "---")
                    if FU:
                        row.append(FU[f"{s}_{t}"][f"{sub}|constrained"]["n_used"])
                    rows.append(row)
                    p = MACRO_SCN[s] + MACRO_TASK[t] + ("Ta" if sub == "task_aware" else "All")
                    M[p + "PrNodes"], M[p + "PrKB"] = i(c["tree_nodes"]), f"{c['c_array_kb']:.1f}"
                    M[p + "PrwF"], M[p + "PrmF"] = pct(c["weighted_f1"]), pct(c["macro_f1"])
                    M[p + "UnNodes"], M[p + "UnKB"] = i(u["tree_nodes"]), f"{u['c_array_kb']:.0f}"
        hdr = ["Scenario", "Task", "Subset", "Feat.", "wF1/mF1", "Nodes", "KB", "Depth/leaf", "wF1/mF1", "Nodes", "KB"]
        if DC:
            hdr.append("Path")
        if FU:
            hdr.append("Used")
        write("tab_pruning.tex", tabular("|l|l|l|r|r|r|r|r|r|r|r|" + ("r|" if DC else "") + ("r|" if FU else ""), hdr, rows))
    if DC:
        rows = []
        for s in SCN:
            for t in TASKS:
                e = DC[f"{s}_{t}"]["selection"]
                rows.append([SCN_LABEL[s], t.capitalize(), e["n"], e["n_K"], i(e["balanced_fold_rows"]),
                             f"{e['fold_seconds']['kmeans']:.1f}", f"{e['fold_seconds']['detector']:.1f}",
                             f"{e['estimated_single_core_selection_s']:.0f}",
                             f"{DC[f'{s}_{t}']['sweep_runtime_s_parallel']:.0f}"])
        write("tab_selection_cost.tex", tabular("|l|l|r|r|r|r|r|r|r|",
              ["Scenario", "Task", "$n$", "Values of $K$", "Fold rows (after SMOTE)", "K-means (s)", "Trees (s)",
               "Single core, total (s)", "Sweep, 5 workers (s)"], rows))
        a = DC["assumptions"]
        M["DcCycles"], M["DcClockMHz"], M["DcPowerW"] = str(a["cycles_per_node"]), f"{a['f_clk_hz'] / 1e6:.0f}", f"{a['p_active_w']:.1f}"
        M["DcNodeBytes"] = str(a["node_bytes"])
        M["DcSelMaxMin"] = f"{max(DC[f'{s}_{t}']['selection']['estimated_single_core_selection_s'] for s in SCN for t in TASKS) / 60:.0f}"
        M["DcCPU"] = tex(DC["machine"]["processor"] or DC["machine"]["platform"])
        for s in SCN:
            for t in TASKS:
                for sub in ["task_aware", "all_features"]:
                    x = DC[f"{s}_{t}"].get(f"{sub}|constrained")
                    if x:
                        p = MACRO_SCN[s] + MACRO_TASK[t] + ("Ta" if sub == "task_aware" else "All")
                        M[p + "Path"] = f"{x['mean_path_length']:.1f}"
                        M[p + "nJ"] = f"{x['estimated_energy_nJ_per_inference']:.0f}"
                        M[p + "VecB"] = str(x["feature_vector_bytes"])

    # ------------------------------------------------------------ evaluation checks (R2.6, R1.12, R1.3)
    if EVC["overlap"] and EVC["blocked"]:
        rows = []
        for s in SCN:
            for t in TASKS:
                o, b = EVC["overlap"][f"{s}_{t}"], EVC["blocked"][f"{s}_{t}"]
                ta = o["task_aware"]
                rows.append([SCN_LABEL[s], t.capitalize(), pct(ta["share_test_vectors_seen_in_train"], 1),
                             f"{ta['seen']['macro_f1']:.3f}" if ta["seen"]["n"] else "---",
                             f"{ta['unseen']['macro_f1']:.3f}" if ta["unseen"]["n"] else "---",
                             pct(o["all_features"]["share_test_vectors_seen_in_train"], 1),
                             f"{b['task_aware']['random_split_weighted_f1']:.3f}/{b['task_aware']['random_split_macro_f1']:.3f}",
                             f"{b['task_aware']['weighted_f1']:.3f}/{b['task_aware']['macro_f1']:.3f}",
                             f"{b['all_features']['weighted_f1']:.3f}/{b['all_features']['macro_f1']:.3f}"])
                p = MACRO_SCN[s] + MACRO_TASK[t]
                M[p + "SeenPct"] = pct(ta["share_test_vectors_seen_in_train"], 1)
                M[p + "SeenAllPct"] = pct(o["all_features"]["share_test_vectors_seen_in_train"], 1)
                M[p + "BlockwF"], M[p + "BlockmF"] = pct(b["task_aware"]["weighted_f1"]), pct(b["task_aware"]["macro_f1"])
                M[p + "BlockAllwF"], M[p + "BlockAllmF"] = pct(b["all_features"]["weighted_f1"]), pct(b["all_features"]["macro_f1"])
        write("tab_independence.tex", tabular("|l|l|r|r|r|r|r|r|r|",
              ["Scenario", "Task", r"Test rows seen on $\mathcal{F}^{*}$ (\%)", "mF1 seen", "mF1 unseen",
               r"Seen, all features (\%)", "Random split", r"Blocked, $\mathcal{F}^{*}$", "Blocked, all"], rows))
    if EVC["confusion"]:
        rows = []
        for pair, e in EVC["confusion"]["pairs"].items():
            a, b = pair.split("|")
            rows.append([f"{a} / {b}", i(e["n_a"]), i(e["n_b"]), e["test_a_pred_b"], e["test_b_pred_a"],
                         i(e["task_aware"]["shared_vectors"]), i(e["all_features"]["shared_vectors"]),
                         f"{e['task_aware'].get('pairwise_cv_balanced_accuracy', float('nan')):.3f}",
                         f"{e['all_features'].get('pairwise_cv_balanced_accuracy', float('nan')):.3f}"])
        TAG = {"Scanning|Reconnaissance": "ScanRec", "DoS|De-authentication": "DosDeauth", "DoS|DDoS": "DosDdos"}
        for pair, e in EVC["confusion"]["pairs"].items():
            t_ = "Conf" + TAG.get(pair, pair.replace("|", "").replace("-", ""))
            M[t_ + "NA"], M[t_ + "NB"] = i(e["n_a"]), i(e["n_b"])
            M[t_ + "SharedAll"] = i(e["all_features"]["shared_vectors"])
            M[t_ + "SharedTa"] = i(e["task_aware"]["shared_vectors"])
            M[t_ + "RowsBShared"] = i(e["all_features"]["rows_b_on_shared_vectors"])
            M[t_ + "BalAll"] = f"{e['all_features'].get('pairwise_cv_balanced_accuracy', float('nan')):.2f}"
            M[t_ + "BalTa"] = f"{e['task_aware'].get('pairwise_cv_balanced_accuracy', float('nan')):.2f}"
        write("tab_s3_confusion.tex", tabular("|l|r|r|r|r|r|r|r|r|",
              ["Pair (A / B)", "Rows A", "Rows B", r"Test A$\to$B", r"Test B$\to$A", r"Shared vectors, $\mathcal{F}^{*}$",
               "Shared vectors, all", r"Pairwise bal.\ acc., $\mathcal{F}^{*}$", "Pairwise bal.\\ acc., all"], rows))
    CEIL = jload(REV, "evaluation_checks", "ceiling.json")
    if CEIL:
        for s in SCN:
            M[MACRO_SCN[s] + "CeilPct"] = pct(CEIL[s]["max_accuracy"], 1)
            M[MACRO_SCN[s] + "ConflictRowsPct"] = pct(CEIL[s]["share_rows_on_conflicting_vectors"], 1)
    if EVC["smote_s1"]:
        e = EVC["smote_s1"]
        for k, lab in [("smote_and_class_weight", "SmoteCw"), ("class_weight_only", "CwOnly"),
                       ("smote_only", "SmoteOnly"), ("neither", "Neither")]:
            M["SOneSmote" + lab + "mF"] = pct(e[k]["macro_f1"])
            M["SOneSmote" + lab + "wF"] = pct(e[k]["weighted_f1"])
            for cname, short in [("MITM", "Mitm"), ("Fake Landing", "Fake"), ("DoS", "Dos")]:
                M["SOneSmote" + lab + short + "Rec"] = f"{e[k]['recall'][cname]:.2f}"

    # ------------------------------------------------------------ duplicate origin (R1.6)
    DO = jload(REV, "duplicate_origin", "summary.json")
    if DO:
        for s in SCN:
            M[MACRO_SCN[s] + "DupAnPct"] = f"{DO[s]['dup_analysis_pct']:.1f}"
            M[MACRO_SCN[s] + "DupRawPct"] = f"{DO[s]['dup_raw_pct']:.1f}"
            M[MACRO_SCN[s] + "DupAdjPct"] = f"{100 * DO[s]['share_of_analysis_dups_adjacent']:.1f}"
        with open(os.path.join(REV, "duplicate_origin", "per_class.csv")) as fh:
            pc = list(csv.DictReader(fh))
        rows = [[SCN_LABEL[r["scenario"]], r["class"], i(r["rows"]), f"{float(r['dup_analysis_pct']):.1f}",
                 f"{float(r['dup_raw_pct']):.1f}"] for r in pc]
        write("tab_dup_origin.tex", tabular("|l|l|r|r|r|",
              ["Scenario", "Class", "Rows", r"Duplicates, analysis columns (\%)", r"Duplicates, all raw columns (\%)"], rows))
