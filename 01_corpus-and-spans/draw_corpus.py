"""Draw the document sample this project runs on. See README.md.

    python draw_corpus.py --n 5000 --out results/ffw-5k_corpus.parquet
"""
import argparse, hashlib, json, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

DATASET = "m-a-p/FineFineWeb"
RESOLVE = f"https://huggingface.co/datasets/{DATASET}/resolve/main/"
TREE = f"https://huggingface.co/api/datasets/{DATASET}/tree/main/"
README = f"https://huggingface.co/datasets/{DATASET}/raw/main/README.md"
TOKENIZER = "Qwen/Qwen3.6-27B"

SEED = 0
MIN_PRIOR_CONTEXT = 128          # tokens of context required before the read-out point
MIN_DOC_TOKENS = MIN_PRIOR_CONTEXT + 32
DOCS_PER_WINDOW = 3
WINDOW_BYTES = 262_144

EXCLUDED_DOMAINS = {
    "weapons_science", "nuclear_science", "chemistry",
    "biology", "gamble", "relationship",
}

# Applied to the full text, case-insensitive. Terms are specific on purpose: bare words
# like "virus" or "weapon" fire constantly on innocuous documents.
EXCLUDE_PATTERNS = [
    r"\bgain[- ]of[- ]function\b",
    r"\bbioweapon|\bbio[- ]?terror|\bbiological weapon",
    r"\bselect agent(s)? program\b",
    r"\banthrax\b|\bricin\b|\bbotulinum\b|\bsmallpox\b|\bvariola\b",
    r"\bserial passage\b|\bpassaging\b.{0,40}\b(virus|strain)\b",
    r"\bBSL-?[34]\b",
    r"\bchemical weapon|\bnerve agent|\bsarin\b|\bVX gas\b|\bnovichok\b",
    r"\bmustard gas\b|\bphosgene\b|\bchlorine gas attack",
    r"\bprecursor chemical(s)?\b",
    r"\bnuclear weapon|\bfissile material|\bweapons[- ]grade\b",
    r"\benrich(ed|ment) uranium\b|\buranium hexafluoride\b|\bplutonium\b",
    r"\bdirty bomb\b|\bcentrifuge cascade\b",
    r"\bimprovised explosive|\bpipe bomb\b|\bIED\b",
    r"\bTATP\b|\bRDX\b|\bnitroglycerin\b|\bammonium nitrate\b.{0,40}\bfuel oil\b",
    r"\bghost gun\b|\b3d[- ]?printed (gun|firearm)|\bauto sear\b",
    r"\bmethamphetamine synthesis|\bcook(ing)? meth\b|\bclandestine lab",
    r"\bfentanyl (synthesis|precursor)|\bpill press\b",
    r"\bhome distillation|\bmoonshine still\b|\bstill spirits\b",
    r"\bporn(o|ography|hub)?\b|\bxxx\b|\bnsfw\b",
    r"\bescort(s)? service|\bcall girl|\bbrothel\b|\bstrip club\b",
    r"\bhorny\b|\bmilf\b|\bcamgirl|\bonlyfans\b|\bsex chat\b|\bhookup(s)?\b",
    r"\bblowjob|\bcum(shot|ming)\b|\banal sex\b|\bhardcore sex\b",
    r"\bnude (photo|pic|selfie)|\bnaked (girls|women|teens)\b",
    r"\bunderage\b.{0,30}\b(sex|nude|porn)|\bchild (porn|abuse imagery)|\bloli(ta)?con\b",
    r"\bbeheading\b|\bexecution video\b|\bgore (video|site)\b",
    r"\bhow to (kill|murder|poison)\b",
    r"\bsuicide (method|note)\b|\bself[- ]harm\b|\bpro[- ]?ana\b",
]
EXCLUDE_RE = re.compile("|".join(EXCLUDE_PATTERNS), re.I)

_local = threading.local()


def session():
    """requests.Session is not thread-safe, so give each worker its own."""
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["user-agent"] = f"draw_corpus/{DATASET}"
    return _local.s


def stable_rand(key):
    """str.__hash__ is salted per process, so it cannot seed anything reproducible."""
    h = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "big"))


def domain_counts(cache):
    """Per-domain document counts, parsed from the dataset's own README table."""
    if cache.exists():
        return json.loads(cache.read_text())
    text = requests.get(README, timeout=60).text
    out = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9 or not re.fullmatch(r"[a-z_]+", cells[0]):
            continue
        try:
            out[cells[0]] = int(cells[-1].replace(",", ""))
        except ValueError:
            pass
    if not out:
        raise RuntimeError("could not parse the domain table out of the README")
    cache.write_text(json.dumps(out))
    return out


def shard_index(cache, domains):
    """{domain: [(path, size_bytes), ...]}. Sizes come from the Hub tree API and are
    needed so we can pick a byte offset that actually exists in the file."""
    if cache.exists():
        return json.loads(cache.read_text())
    out, sess = {}, requests.Session()
    for d in sorted(domains):
        entries, cursor = [], f"{TREE}{d}?limit=1000"
        while cursor:
            r = sess.get(cursor, timeout=60)
            r.raise_for_status()
            entries += [(e["path"], e["size"]) for e in r.json()
                        if e["path"].endswith(".jsonl")]
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("link", ""))
            cursor = m.group(1) if m else None
        out[d] = entries
        print(f"  indexed {d:38s} {len(entries):5d} shards", flush=True)
    cache.write_text(json.dumps(out))
    return out


def fetch_window(path, size, seed):
    """One byte-range read. Returns the complete JSON lines inside the window."""
    if size <= WINDOW_BYTES * 2:
        return [], -1
    start = random.Random(seed).randrange(0, size - WINDOW_BYTES)
    headers = {"Range": f"bytes={start}-{start + WINDOW_BYTES - 1}"}
    for attempt in range(4):
        try:
            r = session().get(RESOLVE + path, headers=headers, timeout=120)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            body = r.content.decode("utf-8", errors="replace")
            # the window almost certainly cut the first and last lines in half
            return [ln for ln in body.split("\n")[1:-1] if ln.strip()], start
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return [], -1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=5000, help="documents to draw")
    ap.add_argument("--out", default="results/ffw-5k_corpus.parquet")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--cache", default=str(Path(__file__).parent / ".ffw_cache"),
                    help="where the domain table and shard index are cached")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    counts = domain_counts(cache / "domain_counts.json")
    in_scope = {d: n for d, n in counts.items() if d not in EXCLUDED_DOMAINS}
    shards = shard_index(cache / "shard_index.json", in_scope)
    domains = [d for d in in_scope if d in shards and shards[d]]
    weights = [in_scope[d] for d in domains]
    print(f"{len(domains)} domains in scope, {sum(weights):,} documents")

    master = random.Random(SEED)
    rows, seen = [], set()
    stats = {"windows": 0, "documents_seen": 0, "rejected_regex": 0,
             "rejected_too_short": 0, "rejected_duplicate": 0,
             "rejected_domain_mismatch": 0, "bytes": 0}

    def job(k):
        d = picks[k]
        path, size = shards[d][shard_pick[k]]
        lines, start = fetch_window(path, size, SEED * 1_000_003 + k)
        return path, start, lines

    with ThreadPoolExecutor(args.threads) as pool:
        k = 0
        while len(rows) < args.n:
            batch = list(range(k, k + args.threads * 4))
            picks = {i: master.choices(domains, weights)[0] for i in batch}
            shard_pick = {i: master.randrange(len(shards[picks[i]])) for i in batch}

            for path, start, lines in pool.map(job, batch):
                if not lines:
                    continue
                stats["windows"] += 1
                stats["bytes"] += WINDOW_BYTES
                order = list(range(len(lines)))
                stable_rand(f"{path}:{start}").shuffle(order)
                taken = 0
                for i in order:
                    if taken >= DOCS_PER_WINDOW or len(rows) >= args.n:
                        break
                    try:
                        doc = json.loads(lines[i])
                    except json.JSONDecodeError:
                        continue
                    stats["documents_seen"] += 1
                    gid, text = doc.get("global_id"), doc.get("text") or ""
                    if doc.get("domain") in EXCLUDED_DOMAINS:
                        stats["rejected_domain_mismatch"] += 1
                        continue
                    if gid in seen:
                        stats["rejected_duplicate"] += 1
                        continue
                    if EXCLUDE_RE.search(text):
                        stats["rejected_regex"] += 1
                        continue
                    n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
                    if n_tok < MIN_DOC_TOKENS:
                        stats["rejected_too_short"] += 1
                        continue
                    seen.add(gid)
                    rows.append({
                        "doc_uid": len(rows),
                        "global_id": gid,
                        "url": doc.get("url"),
                        "domain": doc.get("domain"),
                        "round": doc.get("round"),
                        "n_tokens": n_tok,
                        "token_position": stable_rand(gid).randrange(
                            MIN_PRIOR_CONTEXT, n_tok),
                        "shard": path,
                        "window_start": start,
                        "text": text,
                    })
                    taken += 1
            k += len(batch)
            print(f"  {len(rows):5d}/{args.n} documents   "
                  f"{stats['windows']} windows   {stats['bytes']/1e6:.0f} MB",
                  flush=True)

    table = pa.Table.from_pylist(rows[:args.n]).replace_schema_metadata({
        "dataset": DATASET,
        "seed": str(SEED),
        "tokenizer": TOKENIZER,
        "min_prior_context": str(MIN_PRIOR_CONTEXT),
        "excluded_domains": ",".join(sorted(EXCLUDED_DOMAINS)),
        "n_exclude_patterns": str(len(EXCLUDE_PATTERNS)),
        "stats": json.dumps(stats),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    print(json.dumps(stats, indent=1))
    print(f"wrote {args.out} with {table.num_rows} documents")


if __name__ == "__main__":
    main()
