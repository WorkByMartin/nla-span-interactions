#!/usr/bin/env python3
"""Break the text-space substitution ablation down by the syntactic class of the ablated word.

    python pos_analysis.py --run 3

Prints the class by depth effect table with per-document clustered standard errors,
the extreme individual draws, the sequence-length shares, the leakage-adjusted
effect and the reproduction floor. The same tables are written to
results/statistics.md from the same rendering, and seven figures go to results/.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("RAYON_RS_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import re
import sqlite3
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MLM = "answerdotai/ModernBERT-large"
CANDIDATE_SCHEME = "modernbert-large_filler_model/textsub"
DEPTHS = [1, 3, 5, 10, 25, 50]
OUTLIER_DOC = 2159
MSE_FLOOR = 2e-4
FVE_PER_MSE = 2.1994          # dFVE points per unit dMSE, from the traces variance
T95_39 = 2.02269              # t(0.975) on 39 degrees of freedom, 40 document clusters
MULTITOKEN_CLASSES = ["ADJ", "NOUN", "PROPN", "VERB"]
HIST_LO, HIST_HI, HIST_BINS = -1.0, 1.0, 41

ACCENT = "#4c6ef5"
GREY = "#333333"
DEPTH_RAMP = ["#cdd6fb", "#a9b6f7", "#8595f2", "#6274ec", "#4053c9", "#25327d"]
DEPTH_COLOUR = dict(zip(DEPTHS, DEPTH_RAMP))


# ------------------------------------------------------------------ loading

def connect(path):
    return sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)


def load_draws(conn, run_id):
    """One row per single-substitution draw, with the baseline sequence length attached.

    v_single carries no base_seq_len, and v_baseline is expensive enough that joining
    it row by row is not worth it, so the forty baselines are fetched separately.
    """
    df = pd.read_sql_query(
        """
        SELECT s.variant_id, s.doc_id, s.span_id, s.char_start, s.char_end,
               s.span_text, s.substitute, s.source, s.depth, s.draw_idx, s.prob,
               s.mse, s.fve, s.seq_len, s.base_mse, s.base_fve,
               p.pos
        FROM v_single s
        JOIN v_pos p ON p.span_id = s.span_id
        WHERE s.run_id = ?
        """,
        conn, params=(run_id,))
    base = dict(conn.execute(
        "SELECT doc_id, base_seq_len FROM v_baseline WHERE run_id = ?", (run_id,)))
    df["base_seq_len"] = df["doc_id"].map(base)
    return df


def load_doc_texts(conn, doc_ids):
    """Explanation text per document. Ids are cast, since sqlite3 binds a numpy
    integer to something that matches no row rather than raising."""
    ids = [int(i) for i in doc_ids]
    return dict(conn.execute(
        "SELECT doc_id, text FROM docs WHERE doc_id IN (%s)"
        % ",".join("?" * len(ids)), tuple(ids)))


def load_lexicon(conn):
    """Modal part of speech per lower-cased word type, over every labelled span."""
    rows = conn.execute(
        """
        SELECT lower(trim(t.text)) AS word, l.value AS pos, COUNT(*) AS n
        FROM v_span_text t
        JOIN labels l ON l.span_id = t.span_id AND l.key = 'pos'
                     AND l.scheme LIKE 'spacy-%'
        WHERE trim(t.text) <> ''
        GROUP BY word, pos
        """).fetchall()
    best = {}
    for word, pos, n in rows:
        if word not in best or (n, pos) > best[word]:
            best[word] = (n, pos)
    return {w: p for w, (n, p) in best.items()}


def load_candidates(conn, span_ids):
    out = defaultdict(list)
    for sid, rank, cand in conn.execute(
            f"SELECT span_id, rank, candidate FROM candidates WHERE scheme = ? "
            f"AND rank < {max(DEPTHS)} ORDER BY span_id, rank", (CANDIDATE_SCHEME,)):
        if sid in span_ids:
            out[sid].append(cand)
    return out


def modernbert_token_counts(conn, doc_ids, spans):
    """ModernBERT token count of every span, with the preceding space folded in.

    `spans` maps doc_id to (span_id, char_start, char_end). A span whose characters
    are not tiled exactly by whole ModernBERT tokens gets None.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MLM)
    texts = load_doc_texts(conn, doc_ids)
    counts = {}
    for doc_id, items in spans.items():
        text = texts[doc_id]
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        offs = [(a, b) for a, b in enc["offset_mapping"] if b > a]
        for span_id, start, end in items:
            a = start - 1 if start > 0 and text[start - 1] == " " else start
            hit = [i for i, (x, y) in enumerate(offs) if x < end and y > a]
            ok = hit and offs[hit[0]][0] == a and offs[hit[-1]][1] == end
            counts[span_id] = len(hit) if ok else None
    return counts


# ----------------------------------------------------------------- statistics

def clustered(y, cluster):
    """Mean, cluster-robust standard error, cluster count and draw count."""
    y = np.asarray(y, float)
    n = y.size
    if n == 0:
        return np.nan, np.nan, 0, 0
    m = float(y.mean())
    _, inv = np.unique(np.asarray(cluster), return_inverse=True)
    g = int(inv.max()) + 1
    if g < 2:
        return m, np.nan, g, n
    resid = np.bincount(inv, weights=y - m, minlength=g)
    var = (g / (g - 1)) * float((resid ** 2).sum()) / n ** 2
    return m, float(np.sqrt(var)), g, n


def class_depth_table(df, classes):
    """(class, depth) -> (mean, se, n) of FVE lost, clustered on doc_id."""
    out = {}
    for pos in classes:
        sub = df[df["pos"] == pos]
        for d in DEPTHS:
            cell = sub[sub["depth"] == d]
            m, se, _, n = clustered(cell["fve_lost"], cell["doc_id"])
            out[(pos, d)] = (m, se, n)
    return out


def pooled_by_depth(df):
    """Mean, clustered standard error and draw count per depth, over every class."""
    out = {}
    for d in DEPTHS:
        cell = df[df["depth"] == d]
        m, se, _, n = clustered(cell["fve_lost"], cell["doc_id"])
        out[d] = (m, se, n)
    return out


def purity_by_depth(draws, cands, lexicon):
    """Pooled fraction of the top-k candidates sharing the original's modal class."""
    spans = draws.drop_duplicates("span_id")[["span_id", "pos"]]
    hits = defaultdict(lambda: {d: [0, 0] for d in DEPTHS})
    for span_id, pos in spans.itertuples(index=False):
        cs = cands.get(span_id, [])
        for d in DEPTHS:
            top = cs[:d]
            hit = sum(1 for c in top if lexicon.get(c.strip().lower()) == pos)
            hits[pos][d][0] += hit
            hits[pos][d][1] += len(top)
    return {pos: {d: (v[d][0] / v[d][1] if v[d][1] else np.nan) for d in DEPTHS}
            for pos, v in hits.items()}


# --------------------------------------------------------------------- tables

class Report:
    """Headings, paragraphs and tables built once, rendered as text and as Markdown."""

    def __init__(self):
        self.blocks = []

    def heading(self, text):
        self.blocks.append(("heading", text))

    def para(self, text):
        self.blocks.append(("para", text))

    def table(self, headers, rows):
        self.blocks.append(("table", (list(headers), [list(r) for r in rows])))

    def text(self):
        out = []
        for kind, body in self.blocks:
            if kind == "heading":
                out += ["", body.upper()]
            elif kind == "para":
                out.append(body)
            else:
                headers, rows = body
                w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
                     for i, h in enumerate(headers)]
                line = "  ".join(h.ljust(w[i]) if i == 0 else h.rjust(w[i])
                                 for i, h in enumerate(headers))
                out.append(line)
                out.append("-" * len(line))
                for r in rows:
                    out.append("  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i])
                                         for i, c in enumerate(r)))
        return "\n".join(out)

    def markdown(self, title):
        out = [f"# {title}", ""]
        for kind, body in self.blocks:
            if kind == "heading":
                out += [f"## {body[:1].upper()}{body[1:]}", ""]
            elif kind == "para":
                out += ["\\" + body if body.startswith("*") else body, ""]
            else:
                headers, rows = body
                esc = lambda s: s.replace("|", "\\|")
                out.append("| " + " | ".join(esc(h) for h in headers) + " |")
                out.append("| " + " | ".join("---" if i == 0 else "---:"
                                             for i in range(len(headers))) + " |")
                for r in rows:
                    out.append("| " + " | ".join(esc(c) for c in r) + " |")
                out.append("")
        return "\n".join(out).rstrip() + "\n"


def class_depth_report(rep, df, classes, title):
    tab = class_depth_table(df, classes)
    order = sorted(classes, key=lambda p: -tab[(p, 50)][0])
    rep.heading(title)
    rep.para("mean FVE lost, points, positive = reconstruction got worse; "
             "+/- is the per-document clustered standard error")
    rep.para("* marks a cell whose 95% clustered interval excludes zero")
    rows = []
    for pos in order:
        cells = []
        for d in DEPTHS:
            m, se, _ = tab[(pos, d)]
            star = "*" if se == se and abs(m) > T95_39 * se else ""
            cells.append(f"{m:.3f} +/- {se:.3f}{star}")
        m, se, n = tab[(pos, 50)]
        rows.append([pos, str(n)] + cells
                    + [f"[{m - T95_39 * se:.3f}, {m + T95_39 * se:.3f}]"])
    rep.table(["class", "n@50"] + [f"@{d}" for d in DEPTHS] + ["95% CI @50"], rows)
    sig = [p for p in order if abs(tab[(p, 50)][0]) > T95_39 * tab[(p, 50)][1]]
    rep.para(f"differ from zero at depth 50: {', '.join(sig) if sig else 'none'}")
    return tab


def distinct_substitutes(df):
    """One row per span and distinct substitute string, with that string's FVE lost.

    FVE is a deterministic function of the edited text, so repeated draws of the same
    substitute carry no extra information and are collapsed to one row. A substitute
    that reproduces the original word is an unedited document and is held at exactly
    zero rather than at whatever the repeated forward pass returned.
    """
    d = df.groupby(["span_id", "substitute"], as_index=False).agg(
        fve_lost=("fve_lost", "mean"), n_draws=("fve_lost", "size"),
        depth_first=("depth", "min"), doc_id=("doc_id", "first"),
        span_text=("span_text", "first"), pos=("pos", "first"),
        base_fve=("base_fve", "first"))
    d["is_leak"] = d["substitute"] == d["span_text"]
    d.loc[d["is_leak"], "fve_lost"] = 0.0
    return d


def extreme_spans_report(rep, df, k=20):
    """Spans ranked by their mean over distinct substitutes, not by any single draw."""
    d = distinct_substitutes(df)
    per = d.groupby("span_id").agg(
        doc_id=("doc_id", "first"), span_text=("span_text", "first"),
        pos=("pos", "first"), n_distinct=("substitute", "size"),
        mean_distinct=("fve_lost", "mean"))
    per["n_identical"] = d[d["is_leak"]].groupby("span_id")["n_draws"].sum()
    per["n_identical"] = per["n_identical"].fillna(0).astype(int)
    per["mean_changed"] = d[~d["is_leak"]].groupby("span_id")["fve_lost"].mean()

    for label, asc in [(f"top {k} words by mean FVE lost over distinct substitutes",
                        False),
                       (f"bottom {k} words by mean FVE lost over distinct substitutes",
                        True)]:
        pick = per.sort_values("mean_distinct", ascending=asc)
        pick = pick.reset_index().drop_duplicates(["doc_id", "span_text"]).head(k)
        how = "idxmin" if asc else "idxmax"
        best = d.loc[getattr(d.groupby("span_id")["fve_lost"], how)()]
        best = best.set_index("span_id")
        rep.heading(label)
        rep.para("ranked by the mean over the distinct substitute strings drawn for "
                 "that word, each counted once however many of its 48 draws produced "
                 "it and whatever depth they came from, since FVE is a deterministic "
                 "function of the edited text")
        rep.para("a substitute that reproduces the original word counts once, at an "
                 "FVE lost of exactly zero; n identical is how many of the 48 draws "
                 "were that substitute")
        rep.para(f"the last two columns are the {'smallest' if asc else 'largest'} "
                 "single substitute and its FVE lost, for illustration only; they "
                 "play no part in the ranking")
        rep.para("a word occurring more than once in a document is listed once, at "
                 "its most extreme occurrence")
        rows = []
        for r in pick.itertuples(index=False):
            b = best.loc[r.span_id]
            changed = ("n/a" if r.mean_changed != r.mean_changed
                       else f"{r.mean_changed:.3f}")
            rows.append([str(r.doc_id), r.span_text, r.pos, str(r.n_distinct),
                         str(r.n_identical), f"{r.mean_distinct:.3f}", changed,
                         b.substitute, f"{b.fve_lost:.3f}"])
        rep.table(["doc", "word", "POS", "n distinct", "n identical",
                   "mean over distinct", "mean over distinct non-identical",
                   "extreme substitute", "its FVE lost"], rows)


_MD_SPECIAL = re.compile(r"([\\`*_\[\]~|<>])")


def escape_md(text):
    return _MD_SPECIAL.sub(r"\\\1", text)


def context_window(text, a, b, words=15):
    """The substitution site with about `words` words either side, ellipsis where cut."""
    left = [m.start() for m in re.finditer(r"\S+", text[:a])]
    right = [m.end() + b for m in re.finditer(r"\S+", text[b:])]
    lo = left[-words] if len(left) > words else 0
    hi = right[words - 1] if len(right) >= words else len(text)
    return (" ".join(text[lo:a].split()), " ".join(text[b:hi].split()),
            lo > 0, hi < len(text))


def single_substitutions_report(rep, df, texts, k=5):
    """The most and least costly single substitutions, each shown in its context."""
    d = distinct_substitutes(df)
    d = d.assign(span_mean=d["span_id"].map(d.groupby("span_id")["fve_lost"].mean()))
    span_pos = df.drop_duplicates("span_id").set_index("span_id")
    for label, asc in [(f"largest single substitutions, top {k}", False),
                       (f"largest single substitutions, bottom {k}", True)]:
        rep.heading(label)
        rep.para("each substitution shown in place, the original struck through and "
                 "the substitute in bold, with about 15 words either side; a distinct "
                 "substitute string appears once however many draws produced it")
        for r in d.sort_values("fve_lost", ascending=asc).head(k).itertuples(index=False):
            sp = span_pos.loc[r.span_id]
            before, after, cut_l, cut_r = context_window(
                texts[r.doc_id], sp.char_start, sp.char_end)
            marked = f"~~{escape_md(r.span_text)}~~ **{escape_md(r.substitute)}**"
            pieces = [("... " if cut_l else "") + escape_md(before), marked,
                      escape_md(after) + (" ..." if cut_r else "")]
            rep.para(" ".join(x for x in pieces if x.strip()))
            rep.para(f"doc {r.doc_id}, {r.pos}, depth {r.depth_first}, {r.n_draws} of "
                     f"48 draws, FVE lost {r.fve_lost:.3f}, baseline FVE "
                     f"{r.base_fve:.3f}, word mean over distinct {r.span_mean:.3f}")


def seq_len_report(rep, df, classes):
    delta = df["seq_len"] - df["base_seq_len"]
    n = len(df)
    rep.heading("sequence length: share of substitutions by change in Qwen token count")
    rows, counted = [], 0
    for b in [-2, -1, 0, 1, 2]:
        c = int((delta == b).sum())
        counted += c
        rows.append([f"{b:+d}", str(c), f"{c / n:.4f}"])
    rows.append(["other", str(n - counted), f"{(n - counted) / n:.4f}"])
    rows.append(["total", str(n), f"{1.0:.4f}"])
    rep.table(["change", "n", "share"], rows)

    rep.heading("mean FVE lost, length-preserving vs length-changing, by class")
    keep = delta == 0
    rows = []
    for pos in classes:
        sel = df["pos"] == pos
        a = df.loc[sel & keep, "fve_lost"]
        b = df.loc[sel & ~keep, "fve_lost"]
        da = a.mean() if len(a) else np.nan
        db = b.mean() if len(b) else np.nan
        rows.append([pos, str(len(a)), f"{da:.3f}", str(len(b)), f"{db:.3f}",
                     f"{db - da:.3f}"])
    rep.table(["class", "n same", "mean same", "n changed", "mean changed",
               "difference"], rows)


def leakage_report(rep, df, classes):
    at50 = df[df["depth"] == 50]
    rep.heading("leakage-adjusted effect at depth 50")
    rep.para("leakage = share of draws whose substitute is the original word")
    vals = {}
    for pos in classes:
        sub = at50[at50["pos"] == pos]
        changed = sub[~sub["is_leak"]]
        m_all, _, _, _ = clustered(sub["fve_lost"], sub["doc_id"])
        m_ch, _, _, _ = clustered(changed["fve_lost"], changed["doc_id"])
        vals[pos] = (m_all, float(sub["is_leak"].mean()), m_ch, len(sub), len(changed))
    rep.table(["class", "n", "all draws", "leakage", "n changed", "changed only"],
              [[pos, str(vals[pos][3]), f"{vals[pos][0]:.3f}", f"{vals[pos][1]:.3f}",
                str(vals[pos][4]), f"{vals[pos][2]:.3f}"]
               for pos in sorted(classes, key=lambda p: -vals[p][2])])


def floor_report(rep, df, classes):
    below = df["dmse"].abs() < MSE_FLOOR
    rep.heading("reproduction floor")
    rep.para(f"a per-draw |dMSE| below {MSE_FLOOR:g} is "
             f"{MSE_FLOOR * FVE_PER_MSE * 100:.3f} FVE points, which is the harness "
             f"reproduction floor")
    rep.para(f"overall fraction below the floor: {below.mean():.4f} "
             f"({int(below.sum())} of {len(df)} draws)")
    rep.table(["class", "n", "below floor"],
              [[pos, str(int((df["pos"] == pos).sum())),
                f"{float(below[df['pos'] == pos].mean()):.4f}"] for pos in classes])


def pooled_depth_report(rep, pooled):
    rep.heading("pooled mean FVE lost by depth, over every word class")
    rep.para("clustered on doc_id over 40 documents; the interval is the 95% "
             "document-clustered interval")
    rep.table(["depth", "n", "mean", "clustered SE", "95% CI"],
              [[str(d), str(pooled[d][2]), f"{pooled[d][0]:.3f}",
                f"{pooled[d][1]:.3f}",
                f"[{pooled[d][0] - T95_39 * pooled[d][1]:.3f}, "
                f"{pooled[d][0] + T95_39 * pooled[d][1]:.3f}]"] for d in DEPTHS])


def overflow_report(rep, df, classes):
    over = df["fve_lost"].abs() > HIST_HI
    rep.heading("draws beyond the plotted range")
    rep.para(f"the histogram figures clip FVE lost to plus or minus {HIST_HI:g} points "
             f"and put the overflow in the end bins")
    rep.para(f"overall: {int(over.sum())} of {len(df)} draws overflowed "
             f"({over.mean():.4f})")
    rep.table(["class", "n", "beyond +/- 1", "share"],
              [[pos, str(int((df["pos"] == pos).sum())),
                str(int(over[df["pos"] == pos].sum())),
                f"{float(over[df['pos'] == pos].mean()):.4f}"] for pos in classes])


# -------------------------------------------------------------------- figures

def spread_labels(ys, gap, groups=None):
    """Nudge label positions apart, keeping their order, so text does not overlap.

    `groups` restricts the nudging to points that share a group, so points far
    apart along the other axis are not pushed by each other.
    """
    ys = np.asarray(ys, float)
    groups = np.zeros(len(ys), int) if groups is None else np.asarray(groups)
    out = ys.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        order = idx[np.argsort(ys[idx])]
        for i in range(1, len(order)):
            a, b = order[i - 1], order[i]
            if out[b] - out[a] < gap:
                out[b] = out[a] + gap
    return out


def ridge_panel(ax, values_by_depth, stats_by_depth, edges, step=1.0):
    """One histogram row per depth, each row capped on its own, with a mean and band.

    A row is scaled so that its tallest bin other than the one holding zero reaches
    the top of the row, which leaves the leaked-substitute spike at zero running off
    the top. The vertical line is that row's mean and the shaded band its 95%
    document-clustered interval.
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    middle = (len(edges) - 1) // 2
    height = step * 0.95
    for i, d in enumerate(DEPTHS):
        base = (len(DEPTHS) - 1 - i) * step
        v = np.clip(values_by_depth[d], edges[0], edges[-1] - 1e-12)
        h, _ = np.histogram(v, bins=edges)
        cap = max(float(np.delete(h, middle).max()), 1.0)
        ax.bar(centres, np.minimum(h / cap, 1.0) * height, bottom=base, width=width,
               color=DEPTH_COLOUR[d], linewidth=0)
        ax.axhline(base, color="#bbbbbb", linewidth=0.4)
        m, se = stats_by_depth[d]
        if se == se:
            ax.add_patch(plt.Rectangle((m - T95_39 * se, base), 2 * T95_39 * se, height,
                                       facecolor="#222222", alpha=0.13, linewidth=0))
        ax.plot([m, m], [base, base + height], color="#111111", linewidth=1.0)
    ax.set_yticks([(len(DEPTHS) - 1 - i) * step for i in range(len(DEPTHS))])
    ax.set_yticklabels([f"depth {d}" for d in DEPTHS], fontsize=8, color=GREY)
    ax.set_ylim(-0.12, len(DEPTHS) * step)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(length=2, labelbottom=True)
    for s in ax.spines.values():
        s.set_color("#cccccc")


def hist_edges():
    return np.linspace(HIST_LO, HIST_HI, HIST_BINS + 1)


CAP_RULE = ("each depth row is scaled on its own, so that its tallest bin other than "
            "the one holding zero reaches the top of the row; the bin at zero, which "
            "holds the draws that reproduce the original word, runs off the top")
BAND_RULE = ("the vertical line is that row's mean FVE lost and the shaded band its "
             "95% document-clustered interval")


def fig_fve_lost_by_class(df, classes, tab, path):
    """Ridge line per class: one histogram row per depth, linear axis, mean and band."""
    edges = hist_edges()
    at50 = {p: df.loc[(df["pos"] == p) & (df["depth"] == 50), "fve_lost"].values
            for p in classes}
    order = sorted(classes, key=lambda p: (-np.median(at50[p]), -at50[p].mean()))
    over_all = int((df["fve_lost"].abs() > HIST_HI).sum())

    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.0 * nrow), dpi=200,
                             sharex=True)
    fig.patch.set_facecolor("white")
    axes = np.atleast_1d(axes).ravel()
    for ax, pos in zip(axes, order):
        ax.set_facecolor("white")
        sel = df["pos"] == pos
        vals = {d: df.loc[sel & (df["depth"] == d), "fve_lost"].values for d in DEPTHS}
        stats = {d: (tab[(pos, d)][0], tab[(pos, d)][1]) for d in DEPTHS}
        ridge_panel(ax, vals, stats, edges)
        over = int((df.loc[sel, "fve_lost"].abs() > HIST_HI).sum())
        ax.set_title(f"{pos}   n = {len(at50[pos])} per depth, {over} draws beyond "
                     f"plus or minus 1\nmean at depth 50 = {at50[pos].mean():+.3f} "
                     f"+/- {tab[(pos, 50)][1]:.3f}", fontsize=8, color=GREY)
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.supxlabel(f"FVE lost, points, linear axis clipped to plus or minus 1 with the "
                  f"overflow in the end bins ({over_all} of {len(df)} draws overflowed)",
                  fontsize=9, color=GREY)
    fig.supylabel("bin count, scaled per row", fontsize=9, color=GREY)
    fig.suptitle("Distribution of FVE lost per draw, by word class and candidate "
                 "depth\n" + textwrap.fill(CAP_RULE + "; " + BAND_RULE, 118),
                 fontsize=9.5, color=GREY)
    fig.tight_layout(rect=(0.012, 0.03, 1, 0.925))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_fve_lost_all_classes(df, pooled, path):
    """The same ridge line, pooled over every word class."""
    edges = hist_edges()
    fig, ax = plt.subplots(figsize=(9.5, 6.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    vals = {d: df.loc[df["depth"] == d, "fve_lost"].values for d in DEPTHS}
    ridge_panel(ax, vals, {d: pooled[d][:2] for d in DEPTHS}, edges)
    for i, d in enumerate(DEPTHS):
        m, se, n = pooled[d]
        ax.text(edges[-1] * 0.99, (len(DEPTHS) - 1 - i) + 0.62,
                f"mean {m:+.3f} +/- {se:.3f}, n = {n}", ha="right", fontsize=8,
                color=GREY)
    over_all = int((df["fve_lost"].abs() > HIST_HI).sum())
    ax.set_xlabel(f"FVE lost, points, linear axis clipped to plus or minus 1 with the "
                  f"overflow in the end bins ({over_all} of {len(df)} draws overflowed)")
    ax.set_ylabel("bin count, scaled per row")
    ax.set_title("Distribution of FVE lost per draw, pooled over every word class\n"
                 + textwrap.fill(CAP_RULE + "; " + BAND_RULE, 105),
                 fontsize=9.5, color=GREY)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_mean_by_depth(df, classes, tab, pooled, path):
    """Mean FVE lost against candidate depth, one line per class plus the pooled mean."""
    fig, ax = plt.subplots(figsize=(9.5, 6.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axhline(0, color="#999999", linewidth=0.9)
    x = np.array(DEPTHS, float)
    for p in classes:
        m = np.array([tab[(p, d)][0] for d in DEPTHS])
        e = np.array([T95_39 * tab[(p, d)][1] for d in DEPTHS])
        ax.errorbar(x, m, yerr=e, color=ACCENT, alpha=0.45, linewidth=1.0,
                    marker="o", markersize=3, elinewidth=0.6, capsize=1.5)
    pm = np.array([pooled[d][0] for d in DEPTHS])
    pe = np.array([T95_39 * pooled[d][1] for d in DEPTHS])
    ax.errorbar(x, pm, yerr=pe, color="#111111", linewidth=2.6, marker="o",
                markersize=5, elinewidth=1.6, capsize=4, zorder=5)

    means = np.array([tab[(p, d)][0] for p in classes for d in DEPTHS])
    lo = min(float(means.min()), float((pm - pe).min()))
    hi = max(float(means.max()), float((pm + pe).max()))
    pad = 0.14 * (hi - lo)
    ends = np.array([tab[(p, 50)][0] for p in classes])
    placed = spread_labels(ends, 0.048 * (hi - lo + 2 * pad))
    for p, y0, y1 in zip(classes, ends, placed):
        ax.plot([50, 72], [y0, y1], color="#bbbbbb", linewidth=0.6)
        ax.text(78, y1, p, fontsize=8.5, color=GREY, va="center")

    ax.set_xscale("log")
    ax.set_xticks(DEPTHS)
    ax.set_xticklabels([str(d) for d in DEPTHS])
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.85, 145)
    ax.set_ylim(min(lo - pad, float(placed.min()) - pad / 2),
                max(hi + pad, float(placed.max()) + pad / 2))
    ax.set_xlabel("candidate depth, the number of ranked fillers sampled from")
    ax.set_ylabel("mean FVE lost, points")
    ax.set_title("Does the effect respond to candidate depth?\n"
                 "one line per word class, heavy black line pooled over all classes\n"
                 "vertical bars are 95% document-clustered intervals; the widest class "
                 "intervals run past the axis", fontsize=10.5, color=GREY)
    ax.grid(color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_leakage_purity(df, classes, purity, path):
    leak = {p: [float(df.loc[(df["pos"] == p) & (df["depth"] == d), "is_leak"].mean())
                for d in DEPTHS] for p in classes}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    panels = [(axes[0], leak, "Leakage: share of draws that reproduce the original word"),
              (axes[1], {p: [purity[p][d] for d in DEPTHS] for p in classes},
               "Class purity: share of the top candidates in the original's class")]
    x = np.arange(len(DEPTHS))
    for ax, data, title in panels:
        ax.set_facecolor("white")
        ends = np.array([data[p][-1] for p in classes], float)
        allv = np.array([v for p in classes for v in data[p]], float)
        gap = 0.035 * float(np.nanmax(allv) - np.nanmin(allv))
        placed = spread_labels(ends, gap)
        for p in classes:
            ax.plot(x, data[p], color=ACCENT, linewidth=1.2, alpha=0.75,
                    marker="o", markersize=3)
        for p, y0, y1 in zip(classes, ends, placed):
            ax.plot([x[-1], x[-1] + 0.16], [y0, y1], color="#bbbbbb", linewidth=0.6)
            ax.text(x[-1] + 0.2, y1, p, fontsize=8.5, color=GREY, va="center")
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in DEPTHS])
        ax.set_xlim(-0.2, len(DEPTHS) - 1 + 1.1)
        ax.set_ylim(min(float(np.nanmin(allv)), float(np.nanmin(placed))) - gap,
                    max(float(np.nanmax(allv)), float(np.nanmax(placed))) + gap)
        ax.set_xlabel("candidate depth, the number of ranked fillers sampled from")
        ax.set_title(title, fontsize=10, color=GREY)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color("#cccccc")
    axes[0].set_ylabel("share of draws whose substitute equals the original word")
    axes[1].set_ylabel("share of the top candidates whose modal class matches")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_effect_vs_leakage(df, classes, tab, path):
    at50 = df[df["depth"] == 50]
    xs, ys, es, ns = [], [], [], []
    for p in classes:
        sub = at50[at50["pos"] == p]
        m, se, _ = tab[(p, 50)]
        xs.append(float(sub["is_leak"].mean()))
        ys.append(m)
        es.append(T95_39 * se)
        ns.append(len(sub))
    xs, ys, es, ns = map(np.array, (xs, ys, es, ns))

    fig, ax = plt.subplots(figsize=(9.5, 6.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=ACCENT, elinewidth=1.1,
                capsize=3, alpha=0.8)
    ax.scatter(xs, ys, s=ns / ns.max() * 320 + 25, color=ACCENT, alpha=0.55,
               edgecolor=ACCENT, linewidth=0.8)
    span = float(np.nanmax(ys + es) - np.nanmin(ys - es))
    xorder = np.argsort(xs)
    groups = np.zeros(len(xs), int)
    g, prev = 0, xs[xorder[0]]
    for i in xorder:
        if xs[i] - prev > 0.06:
            g += 1
        groups[i], prev = g, xs[i]
    placed = spread_labels(ys, 0.055 * span, groups)
    label_x = np.array([xs[groups == g].max() + 0.02 for g in groups])
    for p, x0, y0, lx, y1 in zip(classes, xs, ys, label_x, placed):
        ax.annotate(p, (x0, y0), xytext=(lx, y1), fontsize=9, color=GREY,
                    va="center",
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb", linewidth=0.6))
    ax.set_xlabel("leakage at depth 50, the share of draws that reproduce the "
                  "original word")
    ax.set_ylabel("mean FVE lost at depth 50, points, with the 95% "
                  "document-clustered interval")
    ax.set_title("Effect against leakage, marker area proportional to the number of "
                 "draws", fontsize=11, color=GREY)
    ax.set_xlim(xs.min() - 0.05, float(label_x.max()) + 0.07)
    ax.grid(color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#cccccc")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_multitoken(df, path):
    at50 = df[(df["depth"] == 50) & df["n_mlm"].notna()]
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axhline(0, color="#999999", linewidth=0.8)
    splits = [("one ModernBERT token", at50["n_mlm"] == 1, "#b9c4f8"),
              ("more than one ModernBERT token", at50["n_mlm"] > 1, ACCENT)]
    data, positions, colours, notes = [], [], [], []
    for i, pos in enumerate(MULTITOKEN_CLASSES):
        for j, (label, sel, colour) in enumerate(splits):
            v = at50.loc[(at50["pos"] == pos) & sel, "fve_lost"].values
            data.append(v)
            positions.append(i * 1.0 + (j - 0.5) * 0.34)
            colours.append(colour)
            notes.append((positions[-1], len(v)))
    bp = ax.boxplot(data, positions=positions, widths=0.3, whis=(5, 95),
                    showfliers=False, patch_artist=True, medianprops=dict(color="white"))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_edgecolor(c)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set_color("#888888")
    lo = min(float(np.percentile(v, 5)) for v in data if len(v))
    hi = max(float(np.percentile(v, 95)) for v in data if len(v))
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad * 3.2)
    for x, n in notes:
        ax.text(x, hi + pad * 0.7, f"n = {n}", ha="center", fontsize=8, color=GREY)
    ax.set_xticks(range(len(MULTITOKEN_CLASSES)))
    ax.set_xticklabels(MULTITOKEN_CLASSES)
    ax.set_xlim(-0.6, len(MULTITOKEN_CLASSES) - 0.4)
    ax.set_xlabel("word class of the ablated word")
    ax.set_ylabel("FVE lost at depth 50, points")
    ax.set_title("FVE lost at depth 50, split by the ModernBERT token count of the "
                 "original word\nbox is the quartiles, whiskers the 5th and 95th "
                 "percentiles, outliers not drawn", fontsize=10.5, color=GREY)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=c)
               for _, _, c in splits]
    ax.legend(handles, [label for label, _, _ in splits], frameon=False,
              loc="upper right", fontsize=9)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#cccccc")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_per_document(df, path):
    g = df.groupby("doc_id").agg(mean_lost=("fve_lost", "mean"),
                                 base_fve=("base_fve", "first"),
                                 n=("fve_lost", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axhline(0, color="#999999", linewidth=0.9)
    ax.scatter(g["base_fve"], g["mean_lost"], s=42, color=ACCENT, alpha=0.6,
               edgecolor=ACCENT, linewidth=0.8)
    extreme = g.reindex(g["mean_lost"].abs().sort_values(ascending=False).index).head(3)
    for r in extreme.itertuples(index=False):
        ax.annotate(f"document {r.doc_id}", (r.base_fve, r.mean_lost),
                    xytext=(8, 8), textcoords="offset points", fontsize=9, color=GREY,
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb", linewidth=0.6))
    ax.set_xlabel("baseline fraction of variance explained for the unedited document")
    ax.set_ylabel("mean FVE lost across every draw in the document, points")
    ax.set_title("Per-document mean effect against baseline reconstruction quality",
                 fontsize=11, color=GREY)
    ax.grid(color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#cccccc")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="../db/ffw_span-ablation_database.sqlite")
    ap.add_argument("--run", type=int, default=3)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conn = connect(args.db)

    df = load_draws(conn, args.run)
    df = df[~df["source"].str.endswith(" unrecorded")].copy()
    df["fve_lost"] = -100.0 * (df["fve"] - df["base_fve"])
    df["dmse"] = df["mse"] - df["base_mse"]
    df["is_leak"] = df["substitute"] == df["span_text"]
    classes = sorted(df["pos"].unique())

    spans = defaultdict(list)
    for r in df.drop_duplicates("span_id").itertuples(index=False):
        spans[r.doc_id].append((r.span_id, r.char_start, r.char_end))
    counts = modernbert_token_counts(conn, sorted(spans), spans)
    df["n_mlm"] = df["span_id"].map(counts)

    pooled = pooled_by_depth(df)
    doc_texts = load_doc_texts(conn, sorted(df['doc_id'].unique()))
    lexicon = load_lexicon(conn)
    cands = load_candidates(conn, set(df["span_id"].unique()))
    purity = purity_by_depth(df, cands, lexicon)

    rep = Report()
    untiled = sum(1 for v in counts.values() if v is None)
    rep.heading("inputs")
    rep.para(f"run {args.run}: {len(df)} single-substitution draws, "
             f"{df['doc_id'].nunique()} documents, {df['span_id'].nunique()} spans, "
             f"{len(classes)} word classes")
    rep.para(f"ModernBERT token counts: {len(counts) - untiled} spans tiled exactly, "
             f"{untiled} not tiled by whole tokens")
    rep.para(f"corpus lexicon for class purity: {len(lexicon)} word types")

    tab = class_depth_report(rep, df, classes,
                             f"class by depth, all {df['doc_id'].nunique()} documents")
    kept = df[df["doc_id"] != OUTLIER_DOC]
    class_depth_report(rep, kept, classes,
                       f"class by depth, document {OUTLIER_DOC} excluded "
                       f"({kept['doc_id'].nunique()} documents)")
    pooled_depth_report(rep, pooled)
    overflow_report(rep, df, classes)
    extreme_spans_report(rep, df)
    single_substitutions_report(rep, df, doc_texts)
    seq_len_report(rep, df, classes)
    leakage_report(rep, df, classes)
    floor_report(rep, df, classes)

    print(rep.text())
    (out / "statistics.md").write_text(
        rep.markdown(f"Span ablation by syntactic class, run {args.run}"))

    fig_fve_lost_by_class(df, classes, tab, out / "fve_lost_by_class.png")
    fig_fve_lost_all_classes(df, pooled, out / "fve_lost_all_classes.png")
    fig_mean_by_depth(df, classes, tab, pooled, out / "mean_by_depth.png")
    fig_leakage_purity(df, classes, purity, out / "leakage_purity_by_depth.png")
    fig_effect_vs_leakage(df, classes, tab, out / "effect_vs_leakage.png")
    fig_multitoken(df, out / "multitoken_vs_single.png")
    fig_per_document(df, out / "per_document.png")
    print()
    for name in ["statistics.md", "fve_lost_by_class.png",
                 "fve_lost_all_classes.png", "mean_by_depth.png",
                 "leakage_purity_by_depth.png", "effect_vs_leakage.png",
                 "multitoken_vs_single.png", "per_document.png"]:
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
