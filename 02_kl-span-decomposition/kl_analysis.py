"""Where the per-token KL between the RL verbaliser and its SFT reference lands
within a verbalisation, and on what kind of token.

Reads the traces from the project database, prints the loud-token composition,
writes the same report to results/statistics.md, and writes the three figures in results/.

    python kl_analysis.py --db ../db/ffw_span-ablation_database.sqlite --out results
"""

import argparse
import sys
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

TOKENIZER = "Qwen/Qwen3.6-27B"
OPENING_TAG = "<explanation>"
CLOSING_TAG = "</explanation>"
EOS = "<|im_end|>"
BLANK_LINE = re.compile(r"\n[ \t]*\n")

DROP_TAG = True
TRIM_START = 0
TOP_PCT = 1.0
N_FORMS = 30


# --- per-token text --------------------------------------------------------

def bytes_to_unicode():
    """GPT-2 byte-level BPE alphabet: byte value -> the character standing for it."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def token_texts(tok, ids):
    """Per-token text that concatenates back to the exact generated string.

    Recovers each token's raw bytes, decodes the whole byte string once, and
    gives each character to the token holding its first byte.
    """
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    pieces_bytes = []
    for t in tok.convert_ids_to_tokens(ids):
        try:
            pieces_bytes.append(bytes(u2b[c] for c in t))
        except KeyError:
            # Not byte-level (an added token with exotic characters).
            pieces_bytes.append(t.encode("utf-8"))

    blob = b"".join(pieces_bytes)
    text = blob.decode("utf-8", errors="replace")

    char_at_byte = {}
    bpos = 0
    for ci, ch in enumerate(text):
        char_at_byte[bpos] = ci
        bpos += len(ch.encode("utf-8"))

    bounds, bpos = [], 0
    for pb in pieces_bytes:
        bounds.append(bpos)
        bpos += len(pb)
    bounds.append(bpos)

    starts = []
    for b in bounds:
        while b < len(blob) and b not in char_at_byte:
            b += 1
        starts.append(char_at_byte.get(b, len(text)))

    return [text[starts[i]:starts[i + 1]] for i in range(len(pieces_bytes))]


def char_offsets(pieces):
    offs, pos = [], 0
    for p in pieces:
        offs.append(pos)
        pos += len(p)
    return offs


def token_at(offs, char_pos):
    for i in range(len(offs) - 1, -1, -1):
        if offs[i] <= char_pos:
            return i
    return 0


def paragraph_ranges(verb, offs, n_tokens):
    """Token ranges of the blank-line separated blocks, tiling [0, n_tokens)."""
    cuts = [(m.start(), m.end()) for m in BLANK_LINE.finditer(verb)]
    spans, prev = [], 0
    for start, end in cuts:
        spans.append((prev, start))
        prev = end
    spans.append((prev, len(verb)))

    paras, cursor = [], 0
    for i, (c0, c1) in enumerate(spans):
        end_tok = token_at(offs, max(c1 - 1, c0)) + 1
        paras.append({"index": i, "start": cursor, "end": max(end_tok, cursor + 1)})
        cursor = paras[-1]["end"]
    paras[-1]["end"] = max(n_tokens, paras[-1]["start"] + 1)
    return paras


# --- documents -------------------------------------------------------------

def load_docs(db, tok):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    meta = conn.execute(
        "SELECT config FROM runs WHERE script LIKE '%extract_traces.py' "
        "ORDER BY run_id DESC LIMIT 1").fetchone()
    run_config = json.loads(meta[0]) if meta else {}

    rows = conn.execute(
        "SELECT doc_id, domain, verbalisation, mse FROM docs "
        "WHERE verbalisation IS NOT NULL ORDER BY doc_id").fetchall()

    docs = []
    for doc_id, domain, verb, mse in rows:
        trace = conn.execute(
            "SELECT token_id, kl FROM doc_tokens WHERE doc_id = ? ORDER BY position",
            (doc_id,)).fetchall()
        ids = [int(t[0]) for t in trace]
        kl = [float(t[1]) for t in trace]

        pieces = token_texts(tok, ids)
        assert "".join(pieces) == verb + EOS, f"doc {doc_id}: pieces do not rebuild the verbalisation"

        offs = char_offsets(pieces)
        paragraphs = paragraph_ranges(verb, offs, len(ids))
        tag_at = verb.rfind(CLOSING_TAG)
        tag_start = token_at(offs, tag_at) if tag_at >= 0 else len(ids)
        open_at = verb.find(OPENING_TAG)
        body_start = token_at(offs, open_at + len(OPENING_TAG)) if open_at >= 0 else 0

        docs.append({
            "doc_id": doc_id, "domain": domain, "mse": mse,
            "pieces": pieces, "kl": kl, "paragraphs": paragraphs,
            "tag_start": int(tag_start), "body_start": int(min(body_start, tag_start)),
            "n_lines": verb.count("\n") + 1,
        })
    conn.close()
    return docs, run_config


# --- the partition's active range ------------------------------------------

def active_range(doc, drop_tag=DROP_TAG, trim_start=TRIM_START):
    end = doc["tag_start"] if drop_tag else len(doc["kl"])
    head = (doc["body_start"] if drop_tag else 0) + trim_start
    return min(head, max(0, end - 1)), end


def paragraphs_in(doc, rng_):
    start, end = rng_
    out = []
    for p in doc["paragraphs"]:
        s, e = max(p["start"], start), min(p["end"], end)
        if e > s:
            out.append({"index": p["index"], "start": s, "end": e})
    return out


def paragraph_of(idx, paras):
    for p in paras:
        if p["start"] <= idx < p["end"]:
            return p["index"]
    return -1


# --- summary statistics ----------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs)


def quantile(sorted_xs, q):
    if not sorted_xs:
        return float("nan")
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


# --- what carries the highest KL -------------------------------------------

QUOTE = re.compile(r"^[\"'“”‘’]+$")
SENTENCE = re.compile(r"^[.,;:!?]+$")
OTHER_PUNCT = re.compile(r"^[()[\]{}<>/\\|\-–—_*#`~+=@&%$^]+$")
DIGITS = re.compile(r"^[0-9]+$")
LEADING = re.compile(r"^[ \n]")


def classify_token(t):
    """The coarse surface class of one token's text."""
    if "\n" in t:
        return "newline"
    s = t.strip()
    if s == "":
        return "whitespace"
    if QUOTE.match(s):
        return "quote mark"
    if SENTENCE.match(s):
        return "sentence punctuation"
    if OTHER_PUNCT.match(s):
        return "other punctuation"
    if DIGITS.match(s):
        return "digit"
    if not LEADING.match(t):
        return "word continuation"
    return "word start"


def top_token_stats(docs, pct=100 - TOP_PCT, forms=N_FORMS,
                    drop_tag=DROP_TAG, trim_start=TRIM_START):
    rows = []
    for doc in docs:
        rng_ = active_range(doc, drop_tag, trim_start)
        paras = paragraphs_in(doc, rng_)
        last = paras[-1]["index"] if paras else -1
        width = max(1, rng_[1] - rng_[0] - 1)
        for i in range(rng_[0], rng_[1]):
            para = paragraph_of(i, paras)
            rows.append({"kl": doc["kl"][i], "text": doc["pieces"][i],
                         "rel": (i - rng_[0]) / width, "para": para,
                         "in_last": para == last})

    cut = quantile(sorted(r["kl"] for r in rows), pct / 100)
    top = [r for r in rows if r["kl"] >= cut]

    c_top = Counter(classify_token(r["text"]) for r in top)
    c_all = Counter(classify_token(r["text"]) for r in rows)
    classes = sorted(
        [{"name": name, "n": n, "top": n / len(top),
          "all": c_all[name] / len(rows),
          "lift": (n / len(top)) / (c_all[name] / len(rows))}
         for name, n in c_top.items()],
        key=lambda c: -c["top"])

    t_top = Counter(r["text"] for r in top)
    t_all = Counter(r["text"] for r in rows)
    kl_sum = Counter()
    for r in rows:
        kl_sum[r["text"]] += r["kl"]
    # Mean KL then the form itself break the tail of forms tied on loud count
    # and loud rate, so the slice does not depend on document order.
    tokens = sorted(
        [{"text": text, "n": n, "occurrences": t_all[text],
          "loud_rate": n / t_all[text], "mean_kl": kl_sum[text] / t_all[text]}
         for text, n in t_top.items()],
        key=lambda x: (-x["n"], -x["loud_rate"], -x["mean_kl"], x["text"]))[:forms]

    bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
    position = []
    for lo, hi in bins:
        a = sum(1 for r in top if lo <= r["rel"] < hi) / len(top)
        b = sum(1 for r in rows if lo <= r["rel"] < hi) / len(rows)
        position.append({"label": f"{round(lo * 100)}-{round(min(hi, 1) * 100)}%",
                         "top": a, "all": b, "lift": a / b if b else float("nan")})

    where = []
    for label, pred in [
            ("first paragraph", lambda r: r["para"] == 0),
            ("a middle paragraph", lambda r: r["para"] != 0 and not r["in_last"]),
            ("last paragraph", lambda r: r["in_last"])]:
        a = sum(1 for r in top if pred(r)) / len(top)
        b = sum(1 for r in rows if pred(r)) / len(rows)
        where.append({"label": label, "top": a, "all": b,
                      "lift": a / b if b else float("nan")})

    return {
        "percentile": pct, "loud_cut_kl": cut,
        "n_tokens": len(rows), "n_loud": len(top),
        "kl_mean": mean([r["kl"] for r in rows]),
        "kl_max": max(r["kl"] for r in rows),
        "drop_tag": drop_tag, "trim_start": trim_start,
        "classes": classes, "position": position, "paragraph": where,
        "forms": tokens,
    }


# --- figure ----------------------------------------------------------------

def show_token(t):
    t = t.replace("\n", "↵").replace("\t", "→")
    return "␣" + t[1:] if t.startswith(" ") else t


def figure(stats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    forms = stats["forms"][:15][::-1]
    labels = [show_token(f["text"]) for f in forms]
    counts = [f["n"] for f in forms]

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ypos = range(len(forms))
    ax.barh(list(ypos), counts, color="#4c6ef5", height=0.68)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontfamily="monospace", fontsize=9)
    ax.set_xlabel("times the form appears in the loud slice")
    ax.set_title("Most common surface forms among the loudest tokens\n"
                 f"top {100 - stats['percentile']:.0f}% of {stats['n_tokens']:,} tokens by KL",
                 fontsize=11)
    ax.set_xlim(0, max(counts) * 1.42)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)

    for y, f in zip(ypos, forms):
        ax.text(f["n"] + max(counts) * 0.015, y,
                f"{f['n']}/{f['occurrences']}  loud {f['loud_rate'] * 100:.3g}%"
                f"  mean KL {f['mean_kl']:.2f}",
                va="center", fontsize=8, color="#333333")

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# --- figure: one annotated document ----------------------------------------

WRAP = 90
GAMMA = 0.5
FONT_PT = 8.5
CHAR_EM = 0.60205        # DejaVu Sans Mono advance width, in em
LINE_PT = 13.0
MARGIN_COLS = 6
N_BINS = 20


def doc_mean_kl(doc):
    lo, hi = active_range(doc)
    return mean(doc["kl"][lo:hi])


def median_doc(docs):
    """The document whose mean KL sits closest to the median over documents."""
    means = [(doc, doc_mean_kl(doc)) for doc in docs]
    med = quantile(sorted(m for _, m in means), 0.5)
    doc, m = min(means, key=lambda x: (abs(x[1] - med), x[0]["doc_id"]))
    return doc, m, med


def layout_tokens(doc, wrap=WRAP):
    """Place every token of a document on a character grid.

    One cell per drawn run of characters, so a token holding a newline becomes
    several cells. Paragraphs are separated by a blank row.
    """
    cells, col, row = [], 0, 0
    for p in doc["paragraphs"]:
        for i in range(p["start"], min(p["end"], len(doc["pieces"]))):
            for j, seg in enumerate(doc["pieces"][i].split("\n")):
                if j > 0:
                    cells.append({"i": i, "para": p["index"], "text": "↵",
                                  "col": col, "row": row, "width": 1})
                    col, row = 0, row + 1
                if seg == "":
                    continue
                if col and col + len(seg) > wrap:
                    col, row = 0, row + 1
                cells.append({"i": i, "para": p["index"], "text": seg,
                              "col": col, "row": row, "width": len(seg)})
                col += len(seg)
        if col:
            col, row = 0, row + 1
        row += 1

    label_rows = {}
    for c in cells:
        if c["text"] != "↵":
            label_rows.setdefault(c["para"], c["row"])
    n_rows = max((c["row"] for c in cells), default=0) + 1
    return cells, label_rows, n_rows


def document_figure(doc, mean_kl, path):
    """Draw one verbalisation with every token shaded by its KL."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle

    lo, hi = active_range(doc)
    norm = PowerNorm(gamma=GAMMA, vmin=0.0, vmax=max(doc["kl"][lo:hi]))
    cmap = plt.get_cmap("Blues")

    cells, label_rows, n_rows = layout_tokens(doc)
    char_w = CHAR_EM * FONT_PT
    ax_w = (WRAP + MARGIN_COLS) * char_w / 72
    ax_h = n_rows * LINE_PT / 72
    left, right, top, bottom = 0.45, 0.35, 1.0, 1.0
    fig = plt.figure(figsize=(left + right + ax_w, top + bottom + ax_h), dpi=200)
    fig.patch.set_facecolor("white")
    figw, figh = fig.get_size_inches()
    ax = fig.add_axes([left / figw, bottom / figh, ax_w / figw, ax_h / figh])
    ax.set_facecolor("white")
    ax.set_xlim(-MARGIN_COLS, WRAP)
    ax.set_ylim(n_rows, 0)
    ax.axis("off")

    for c in cells:
        i = c["i"]
        if lo <= i < hi:
            face = cmap(norm(doc["kl"][i]))
            lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
            ink = "#ffffff" if lum < 0.5 else "#111111"
        else:
            face, ink = "#eeeeee", "#999999"
        ax.add_patch(Rectangle((c["col"], c["row"] + 0.08), c["width"], 0.84,
                               facecolor=face, edgecolor="none", zorder=1))
        ax.text(c["col"] + 0.06, c["row"] + 0.5, c["text"], ha="left", va="center",
                fontname="DejaVu Sans Mono", fontsize=FONT_PT, color=ink, zorder=2)

    for para, row in sorted(label_rows.items()):
        ax.text(-MARGIN_COLS + 0.4, row + 0.5, f"¶{para + 1}",
                ha="left", va="center", fontname="DejaVu Sans Mono",
                fontsize=FONT_PT, color="#777777")

    fig.text(0.5, 1 - 0.34 / figh,
             f"Token by token KL for document {doc['doc_id']}",
             ha="center", va="top", fontsize=11)
    fig.text(0.5, 1 - 0.56 / figh,
             f"{len(doc['paragraphs'])} paragraphs, mean KL {mean_kl:.4f}; "
             "the closing tag is greyed and excluded from the shading range",
             ha="center", va="top", fontsize=9, color="#444444")

    cax = fig.add_axes([0.30, 0.42 / figh, 0.40, 0.13 / figh])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=cax, orientation="horizontal")
    bar.set_label("KL divergence at the token (shading uses a square root scale)",
                  fontsize=8)
    bar.ax.tick_params(labelsize=7)

    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# --- figure: KL against position -------------------------------------------

def position_profile(docs, cut, n_bins=N_BINS, drop_tag=DROP_TAG, trim_start=TRIM_START):
    """Per-token KL and loud-token counts, binned by relative position."""
    kls = [[] for _ in range(n_bins)]
    loud = [0] * n_bins
    n_loud = 0
    for doc in docs:
        rng_ = active_range(doc, drop_tag, trim_start)
        width = max(1, rng_[1] - rng_[0] - 1)
        for i in range(rng_[0], rng_[1]):
            b = min(int((i - rng_[0]) / width * n_bins), n_bins - 1)
            kls[b].append(doc["kl"][i])
            if doc["kl"][i] >= cut:
                loud[b] += 1
                n_loud += 1
    bins = []
    for b in range(n_bins):
        xs = sorted(kls[b])
        bins.append({"lo": 100 * b / n_bins, "hi": 100 * (b + 1) / n_bins,
                     "n": len(xs), "mean": mean(xs) if xs else float("nan"),
                     "median": quantile(xs, 0.5),
                     "q25": quantile(xs, 0.25), "q75": quantile(xs, 0.75),
                     "loud": loud[b],
                     "loud_share": loud[b] / n_loud if n_loud else 0.0})
    return {"n_bins": n_bins, "n_loud": n_loud, "cut": cut, "bins": bins}


def position_figure(prof, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = prof["bins"]
    centres = [(b["lo"] + b["hi"]) / 2 for b in bins]
    width = bins[0]["hi"] - bins[0]["lo"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), dpi=200, sharex=True,
                                   gridspec_kw={"height_ratios": [1.25, 1]})
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)

    ax1.fill_between(centres, [b["q25"] for b in bins], [b["q75"] for b in bins],
                     color="#4c6ef5", alpha=0.18,
                     label="25th to 75th percentile of the tokens in the bin")
    ax1.plot(centres, [b["median"] for b in bins], color="#4c6ef5", linewidth=1.8,
             marker="o", markersize=3.5, label="median KL divergence in the bin")
    ax1.plot(centres, [b["mean"] for b in bins], color="#c2410c", linewidth=1.4,
             linestyle="--", label="mean KL divergence in the bin (heavy tail pulls it above the band)")
    ax1.set_ylabel("KL divergence per token")
    ax1.set_title("KL divergence along the generation\n"
                  f"{prof['n_bins']} equal width position bins, closing tag excluded",
                  fontsize=11)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.set_ylim(bottom=0)

    ax2.bar(centres, [b["loud_share"] for b in bins], width=width * 0.85,
            color="#e8590c",
            label=f"share of the {prof['n_loud']} loud tokens landing in the bin")
    ax2.axhline(1 / prof["n_bins"], color="#333333", linestyle="--", linewidth=1.1,
                label=f"uniform expectation, 1 bin in {prof['n_bins']}")
    ax2.set_ylabel("share of the loud tokens")
    ax2.set_xlabel("position in the generation (percent of the tokens scored)")
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.set_xlim(0, 100)
    ax2.set_xticks(list(range(0, 101, 10)))

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# --- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="../db/ffw_span-ablation_database.sqlite")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--out", default="results", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    docs, run_config = load_docs(args.db, tok)

    class Tee:
        """Write the report to stdout and to results/statistics.md at once."""
        def __init__(self, path):
            self.f = open(path, "w")
        def write(self, s):
            sys.__stdout__.write(s)
            self.f.write(s)
        def flush(self):
            sys.__stdout__.flush()
            self.f.flush()
    sys.stdout = Tee(args.out / "statistics.md")
    sys.stdout.f.write("```\n")

    lift = top_token_stats(docs)
    figure(lift, args.out / "kl_token_frequency.png")
    example, example_kl, median_kl = median_doc(docs)
    document_figure(example, example_kl, args.out / "kl_example_document.png")
    profile = position_profile(docs, lift["loud_cut_kl"])
    position_figure(profile, args.out / "kl_by_position.png")

    md = run_config.get("metadata", {})
    fve_baseline = float(md["fve_baseline"]) if "fve_baseline" in md else None
    mses = [d["mse"] for d in docs if d["mse"] is not None]
    para_hist = Counter(len(d["paragraphs"]) for d in docs)

    print("SAMPLE")
    print(f"  documents {len(docs)}, generated tokens {sum(len(d['kl']) for d in docs)}")
    print("  paragraphs per document: "
          + ", ".join(f"{k}: {para_hist[k]}" for k in sorted(para_hist)))
    if mses and fve_baseline:
        print(f"  mse mean {mean(mses):.4f}, fve mean {1 - mean(mses) / fve_baseline:.4f} "
              f"(baseline {fve_baseline:.4f})")
    print(f"  kl mean {lift['kl_mean']:.4f}, kl max {lift['kl_max']:.4f}")

    print(f"\nLOUD SLICE: tokens above the {lift['percentile']}th percentile of KL, "
          f"{lift['n_loud']} of {lift['n_tokens']}, KL >= {lift['loud_cut_kl']:.4f}")
    print(f"  {'surface class':22s} {'loud':>6s} {'all':>7s} {'lift':>6s}   (shares of the loud slice and of all tokens)")
    for c in sorted(lift["classes"], key=lambda c: -c["lift"]):
        print(f"  {c['name']:22s} {c['top']:6.3f} {c['all']:7.3f} {c['lift']:6.2f}")
    for section, rows in (("position quintile", lift["position"]), ("paragraph", lift["paragraph"])):
        print(f"  {section:22s} {'loud':>6s} {'all':>7s} {'lift':>6s}")
        for r in rows:
            print(f"    {str(r['label']):20s} {r['top']:6.3f} {r['all']:7.3f} {r['lift']:6.2f}")
    print(f"  {'surface form':22s} {'loud':>6s} {'seen':>7s} {'rate':>6s} {'mean kl':>8s}")
    for f in lift["forms"][:30]:
        print(f"  {repr(f['text']):22s} {f['n']:6d} {f['occurrences']:7d} {f['loud_rate']:6.2f} {f['mean_kl']:8.2f}")

    print("\nFIGURES")
    print(f"  kl_example_document.png: document {example['doc_id']}, every token shaded "
          f"by its KL, paragraph breaks marked; chosen for its mean KL "
          f"{example_kl:.4f}, nearest the median over documents of {median_kl:.4f}")
    print(f"  kl_by_position.png: median KL with the 25th to 75th percentile band, and the mean, over "
          f"{profile['n_bins']} position bins, and the share of the {profile['n_loud']} "
          f"loud tokens falling in each bin")
    sys.stdout.f.write("```\n")
    sys.stdout.flush()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
