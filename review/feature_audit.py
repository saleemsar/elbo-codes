"""Feature validity audit (reviewer comments R1.7 and R2.5).

The UAV-NIDD CSV headers are not reliably aligned with their values: many columns
hold values that are impossible for the field named in the header (for example a
1-bit radiotap flag holding capture times of up to 3,122 s, or `uid` holding IPv4
addresses). This script checks every raw column against the documented value
domain of its header field and classifies it as

  identifier  content is an address, a port, an absolute or capture-relative
              timestamp, a hardware clock (TSF) or a frame/sequence counter
              (by content OR by header), so it can identify the capture session
              instead of describing traffic;
  validated   >= VALID_MIN of the populated values lie in the header's domain, the
              column is not an identifier, and (S3) the arithmetic identities of its
              statistic family hold (min <= avg <= max, ...);
  invalid     everything else (out-of-domain values, mixed content, non-numeric
              protocol strings).

Zero is the dataset's fill value for absent protocol fields, so zeros are counted
as "absent" for protocol-specific fields and as out-of-domain for fields every
frame must have (frame.len, frame.encap_type, ...).

Domains follow the Wireshark display-filter reference (fields of frame, radiotap,
wlan, wlan_radio, eapol, arp, ip, tcp, udp, icmpv6, http) and the Zeek conn.log /
flow-meter field definitions for S3.

S3 is audited after its one-column header offset is corrected (uav_ehfs.s3_realigned_names);
s3_alignment_check() documents the arithmetic identities that establish that offset.

Usage:  python review/feature_audit.py
Output: review/feature_audit/<S>_columns.csv, review/feature_audit/summary.json
        (summary.json is read by uav_ehfs.py --features validated)
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import uav_ehfs as U  # noqa: E402

OUT = os.path.join(HERE, "feature_audit")
VALID_MIN = 0.99        # share of populated values that must be in domain
ID_CONTENT_MIN = 0.01   # share of values that look like addresses / clocks to flag an identifier
ORDER_RHO = 0.90        # |Spearman(value, row order)|, over the file or within each capture session,
                        # above which a column is treated as a clock or counter

_MAC_RE = re.compile(r"^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$")

INF = np.inf
# (kind, lo, hi, integer, zero_is_absent)
#   kind: "num" checked by range; "id" identifier by definition; "str" not a numeric field
BIT = ("num", 0, 1, True, False)
D = {
    # ---------------- frame
    "frame.encap_type": ("num", 1, 250, True, False),
    "frame.len": ("num", 14, 65535, True, False),
    "frame.number": ("id",),
    "frame.time_delta_displayed": ("num", 0, 1e4, False, False),
    "frame.time_epoch": ("id",),
    "frame.time_relative": ("id",),
    # ---------------- radiotap
    "radiotap.channel.flags.cck": BIT,
    "radiotap.channel.flags.ofdm": BIT,
    "radiotap.channel.freq": ("num", 2412, 5900, True, True),
    "radiotap.datarate": ("num", 1, 600, False, True),
    "radiotap.dbm_antsignal": ("num", -120, -1, True, True),
    "radiotap.length": ("num", 8, 128, True, True),
    "radiotap.mactime": ("id",),
    "radiotap.present.tsft": BIT,
    "radiotap.rxflags": ("num", 0, 65535, True, True),
    "radiotap.timestamp.ts": ("id",),
    "radiotap.vendor_oui": ("num", 0, 2 ** 24 - 1, True, True),
    # ---------------- wlan
    "wlan.duration": ("num", 0, 32767, True, True),
    "wlan.bssid": ("id",),
    "wlan.fc.retry": BIT,
    "wlan.fc.moredata": BIT,
    "wlan.fc.protected": BIT,
    "wlan.fc.pwrmgt": BIT,
    "wlan.fc.type": ("num", 0, 3, True, False),
    "wlan.fc.subtype": ("num", 0, 15, True, False),
    "wlan.fcs.bad_checksum": BIT,
    "wlan.fixed.beacon": ("num", 1, 65535, False, True),
    "wlan.fixed.capabilities.ess": BIT,
    "wlan.seq": ("id",),
    "wlan.tag": ("str",),
    "wlan_radio.frequency": ("num", 2412, 5900, True, True),
    "wlan_radio.signal_dbm": ("num", -120, -1, True, True),
    "wlan_radio.start_tsf": ("id",),
    "wlan_radio.end_tsf": ("id",),
    "wlan_radio.timestamp": ("id",),
    "wlan_radio.phy": ("num", 1, 15, True, True),
    "wlan_radio.data_rate": ("num", 1, 600, False, True),
    "wlan.rsn.capabilities.mfpc": BIT,
    "wlan_rsna_eapol.keydes.msgnr": ("num", 1, 4, True, True),
    "eapol.keydes.replay_counter": ("num", 0, 2 ** 64, True, True),
    # ---------------- arp / ip / transport
    "arp": ("str",),
    "arp.hw.type": ("num", 1, 1024, True, True),
    "arp.proto.type": ("num", 1, 65535, True, True),
    "ip.dst": ("id",),
    "ip.src": ("id",),
    "ip.proto": ("num", 0, 255, True, True),
    "ip.ttl": ("num", 1, 255, True, True),
    "tcp.ack": ("num", 0, 2 ** 32 - 1, True, True),
    "udp.dstport": ("id",),
    "udp.srcport": ("id",),
    "udp.length": ("num", 8, 65535, True, True),
    "data.len": ("num", 1, 65535, True, True),
    "icmpv6.mldr.nb_mcast_records": ("num", 0, 65535, True, True),
    "nbns": ("str",),
    "nbss.continuation_data": ("str",),
    "dns": ("str",),
    "http.connection": ("str",),
    "http.content_length": ("num", 0, 2 ** 31, True, True),
    "http.content_type": ("str",),
    "http.date": ("str",),
    "http.file_data": ("str",),
    "http.host": ("str",),
    "http.last_modified": ("str",),
    "http.location": ("str",),
    # ---------------- S3: Zeek conn.log identifiers
    "uid": ("id",),
    "id.orig_h": ("id",),
    "id.orig_p": ("id",),
    "id.resp_h": ("id",),
    "id.resp_p": ("id",),
    # S3 columns without a matching header after realignment (uav_ehfs.s3_realigned_names)
    "unverified_bytes_col4": ("unverified",),
    "unverified_window_col83": ("unverified",),
}
NONNEG = ("num", 0, INF, False, False)
NONNEG_INT = ("num", 0, INF, True, False)
WINDOW = ("num", 0, 65535, True, False)


def domain(col):
    if col in D:
        return D[col]
    # S3 flow-meter statistics: non-negative; counts are integers
    if col.endswith("_window_size"):
        return WINDOW
    if col.endswith("_flag_count") or col.endswith("pkts_tot") or col.endswith("header_size_tot") \
            or col.endswith("header_size_min") or col.endswith("header_size_max"):
        return NONNEG_INT
    return NONNEG


def within_run_rho(num, runs):
    """|Spearman(value, row order)| inside each contiguous label run (= one capture
    session), weighted by run size: close to 1 for a clock or counter that restarts
    with every capture file."""
    tot, acc = 0, 0.0
    for r in np.unique(runs):
        m = (runs == r) & num.notna().values & (num.values != 0)
        k = int(m.sum())
        if k < 50 or np.unique(num.values[m]).size < 10:
            continue
        rho = spearmanr(np.flatnonzero(m), num.values[m]).statistic
        if np.isfinite(rho):
            acc += k * abs(rho)
            tot += k
    return float(acc / tot) if tot else 0.0


def content_profile(raw, num, idx, runs):
    """What the values look like, independent of the header."""
    s = raw.dropna().astype(str).str.strip()
    n = max(len(s), 1)
    nz = num[(num != 0) & num.notna()]
    p = {
        "frac_ip": float(s.str.match(U._IP_RE).sum() / n),
        "frac_mac": float(s.str.match(_MAC_RE).sum() / n),
        "frac_nonnumeric": float((num.isna() & raw.notna()).sum() / max(raw.notna().sum(), 1)),
        "frac_epoch_s": float(((nz >= 1e9) & (nz <= 2e9)).sum() / n),
        "frac_clock_us": float((nz > 1e10).sum() / n),
        "frac_zero": float((num == 0).sum() / n),
        "n_unique": int(s.nunique()),
        "min": float(num.min()) if num.notna().any() else None,
        "max": float(num.max()) if num.notna().any() else None,
    }
    m = num.notna() & (num != 0)
    p["rho_row_order"] = float(spearmanr(idx[m.values], num[m].values).statistic) if m.sum() > 100 else 0.0
    p["rho_within_capture"] = within_run_rho(num, runs)
    top = s.value_counts().index[:4].tolist()
    p["examples"] = " | ".join(top)
    return p


def in_domain(num, dom):
    kind, lo, hi, integer, zero_absent = dom
    v = num.copy()
    if zero_absent:
        v = v[v != 0]
    pop = v.notna().sum() + (num.isna().sum() if not zero_absent else 0)
    ok = v.notna() & (v >= lo) & (v <= hi)
    if integer:
        ok &= (v == np.round(v))
    return float(ok.sum() / pop) if pop else 1.0, int(pop)


FAMILY_RE = re.compile(r"^(.*)\.(min|max|avg|tot|std)$")


def family_checks(Xn):
    """S3: row-wise arithmetic identities of each statistic family (min<=avg<=max<=tot, std>=0).
    Only rows on which the family carries information (some member non-zero) are
    counted, because all-zero rows satisfy every identity trivially.
    Returns {column: share of informative rows on which every identity involving it holds}."""
    fam = {}
    for c in Xn.columns:
        m = FAMILY_RE.match(c)
        if m:
            fam.setdefault(m.group(1), {})[m.group(2)] = c
    res = {}
    for base, cols in fam.items():
        pairs = [(a, b) for a, b in [("min", "avg"), ("avg", "max"), ("min", "max"), ("max", "tot")]
                 if a in cols and b in cols]
        info = (Xn[list(cols.values())].fillna(0) != 0).any(axis=1)
        for stat, c in cols.items():
            okc = pd.Series(True, index=Xn.index)
            for a, b in pairs:
                if stat in (a, b):
                    okc &= Xn[cols[a]] <= Xn[cols[b]] * (1 + 1e-9) + 1e-9
            res[c] = float(okc[info].mean()) if info.any() else 1.0
    return res


def audit(scn):
    df, label_col = U.load_raw(scn)   # S3 with realigned headers (uav_ehfs.s3_realigned_names)
    X = df.drop(columns=[label_col])
    idx = np.arange(len(df))
    y = df[label_col].map(U.canon_label)
    runs = (y != y.shift()).cumsum().to_numpy()
    Xn = pd.DataFrame({c: U.parse_numeric(X[c]) for c in X.columns})
    fam = family_checks(Xn) if scn == "S3" else {}
    rows = []
    for c in X.columns:
        dom = domain(c)
        prof = content_profile(X[c], Xn[c], idx, runs)
        is_id_content = (prof["frac_ip"] + prof["frac_mac"] >= ID_CONTENT_MIN
                         or prof["frac_epoch_s"] + prof["frac_clock_us"] >= ID_CONTENT_MIN
                         or abs(prof["rho_row_order"]) >= ORDER_RHO
                         or prof["rho_within_capture"] >= ORDER_RHO)
        r = {"scenario": scn, "column": c, "header_domain": dom[0] if dom[0] != "num" else
             f"[{dom[1]:g}, {dom[2]:g}]{' int' if dom[3] else ''}{' (0=absent)' if dom[4] else ''}"}
        r.update(prof)
        reasons = []
        if dom[0] == "id":
            reasons.append("identifier by header")
        if is_id_content:
            reasons.append("identifier/clock content")
        if dom[0] == "num":
            share, pop = in_domain(Xn[c], dom)
            r["in_domain_share"] = share
            r["n_populated"] = pop
        else:
            r["in_domain_share"] = None
            r["n_populated"] = int(X[c].notna().sum())
        if c in fam:
            r["family_identity_share"] = fam[c]
        if reasons:
            r["status"] = "identifier"
        elif dom[0] == "str":
            r["status"] = "invalid"
            reasons.append("protocol string, not a numeric field")
        elif dom[0] == "unverified":
            r["status"] = "invalid"
            reasons.append("no header matches this column")
        elif r["in_domain_share"] < VALID_MIN:
            r["status"] = "invalid"
            reasons.append(f"only {100 * r['in_domain_share']:.1f}% of populated values in header domain")
        elif c in fam and fam[c] < VALID_MIN:
            r["status"] = "invalid"
            reasons.append(f"statistic-family identities hold on {100 * fam[c]:.1f}% of rows")
        else:
            r["status"] = "validated"
        r["reason"] = "; ".join(reasons)
        rows.append(r)
    return pd.DataFrame(rows)


def s3_alignment_check():
    """Share of informative S3 rows satisfying flow-statistic identities when the shipped
    header is read as is (offset 0) and when value column i is read as header i+1
    (offset 1, the realignment used by uav_ehfs.py)."""
    saved = U.FEATURE_SET
    U.FEATURE_SET = "published"          # load the shipped header
    df, lab = U.load_raw("S3")
    U.FEATURE_SET = saved
    H = [c for c in df.columns if c != lab]
    V = [U.parse_numeric(df[c]) for c in H]

    def close(a, b, rel=0.02):
        return np.abs(a - b) <= rel * np.maximum(np.abs(a), np.abs(b)) + 1e-6

    rows = []
    for off in (0, 1):
        R = {H[i + off]: V[i] for i in range(len(H)) if 0 <= i + off < len(H)}

        def info(*cs):
            return (pd.concat([R[c] for c in cs], axis=1).fillna(0) != 0).any(axis=1)

        checks = {}
        for fam in ["fwd_pkts_payload", "bwd_pkts_payload", "flow_pkts_payload",
                    "fwd_iat", "bwd_iat", "flow_iat", "active", "idle"]:
            mn, av, mx, tt = [R[f"{fam}.{k}"] for k in ["min", "avg", "max", "tot"]]
            m = info(f"{fam}.min", f"{fam}.max", f"{fam}.tot")
            ok = (mn <= av * (1 + 1e-9) + 1e-9) & (av <= mx * (1 + 1e-9) + 1e-9) & (mx <= tt * (1 + 1e-9) + 1e-9)
            checks[f"{fam}: min <= avg <= max <= tot"] = (ok[m].mean(), int(m.sum()))
        for d in ["fwd", "bwd"]:
            m = info(f"{d}_pkts_payload.tot")
            ok = close(R[f"{d}_pkts_payload.tot"], R[f"{d}_pkts_payload.avg"] * R[f"{d}_pkts_tot"])
            checks[f"{d}_pkts_payload.tot = avg x {d}_pkts_tot"] = (ok[m].mean(), int(m.sum()))
        m = info("flow_pkts_payload.tot")
        ok = close(R["flow_pkts_payload.tot"], R["fwd_pkts_payload.tot"] + R["bwd_pkts_payload.tot"])
        checks["flow_pkts_payload.tot = fwd + bwd"] = (ok[m].mean(), int(m.sum()))
        m = R["flow_iat.tot"] > 0
        dur = R["flow_iat.tot"] / 1e6
        ok = close(R["flow_pkts_per_sec"], (R["fwd_pkts_tot"] + R["bwd_pkts_tot"]) / dur, rel=0.05)
        checks["flow_pkts_per_sec = packets / duration"] = (ok[m].mean(), int(m.sum()))
        ok = close(R["payload_bytes_per_second"], R["flow_pkts_payload.tot"] / dur, rel=0.05)
        checks["payload_bytes_per_second = payload / duration"] = (ok[m].mean(), int(m.sum()))
        for k, (share, n) in checks.items():
            rows.append({"identity": k, "offset": off, "share_rows_holding": float(share), "n_rows": n})
    t = pd.DataFrame(rows).pivot(index="identity", columns="offset", values="share_rows_holding")
    t.columns = ["as_shipped", "realigned"]
    n = pd.DataFrame(rows).query("offset == 1").set_index("identity")["n_rows"]
    t["n_rows"] = n
    return t.reset_index()


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {"VALID_MIN": VALID_MIN, "ID_CONTENT_MIN": ID_CONTENT_MIN, "ORDER_RHO": ORDER_RHO}
    al = s3_alignment_check()
    al.to_csv(os.path.join(OUT, "S3_alignment.csv"), index=False)
    print(al.to_string())
    summary["S3_alignment"] = al.to_dict(orient="records")
    for scn in ["S1", "S2", "S3"]:
        t = audit(scn)
        t.to_csv(os.path.join(OUT, f"{scn}_columns.csv"), index=False)
        summary[scn] = {s: sorted(t.loc[t.status == s, "column"]) for s in ["validated", "invalid", "identifier"]}
        print(f"[{scn}] " + ", ".join(f"{s}: {len(v)}" for s, v in summary[scn].items()))
        print("   validated:", summary[scn]["validated"])
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
