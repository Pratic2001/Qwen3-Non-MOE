"""
quality.py

Shared quality-filtering / dedup / shard-writing primitives, factored out of
build_dataset.py so the live-scraping agent applies EXACTLY the same bar as
the HF-streaming pipeline. If you tighten a filter here, both pipelines get
the improvement.
"""

import hashlib
import json
import os
import re
from typing import Optional

SHARD_MAX_BYTES = 256 * 1024 * 1024  # 256MB per shard file, matches build_dataset.py

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Boilerplate / nav-junk phrases that show up disproportionately in scraped
# (as opposed to pre-filtered HF) web text. Cheap substring check, applied
# before the heavier heuristics.
_JUNK_MARKERS = (
    "enable javascript", "cookies to continue", "subscribe to continue",
    "404 not found", "access denied", "please verify you are a human",
    "add to cart", "sign in to your account", "captcha",
)


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text)


def _top_word_repetition_ratio(text: str) -> float:
    words = _WORD_RE.findall(text.lower())
    if len(words) < 10:
        return 0.0
    counts: dict = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return max(counts.values()) / len(words)


def passes_prose_quality_filter(text: str, min_doc_chars: int = 500) -> bool:
    if len(text) < min_doc_chars:
        return False
    if _alpha_ratio(text) < 0.6:
        return False
    if _top_word_repetition_ratio(text) > 0.30:
        return False
    lines = text.split("\n")
    if len(lines) < 3 and len(text) > 2000:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _JUNK_MARKERS):
        return False
    return True


def passes_sft_pair_quality_filter(prompt: str, answer: str, min_chars: int = 20) -> bool:
    """Quality bar for a (prompt, answer) pair that the source dataset
    already labeled itself (an HF instruction dataset, a Kaggle Q&A CSV,
    ...), as opposed to raw scraped prose. These are semantically complete
    even when short -- a math problem + a short numeric answer, a one-line
    code question + a one-line fix -- so this deliberately does NOT reuse
    passes_prose_quality_filter's 500-char default floor, which is tuned
    for scraped web articles and rejects almost all short Q&A pairs
    (an entire math/code SFT dataset can come back 100% REJECTed against
    that bar, well before anything reaches the LLM judge).

    Checks that matter for a labeled pair instead:
    - both sides present and non-trivial (not just whitespace/punctuation)
    - combined length clears a much lower floor than prose (min_chars)
    - not degenerate repetition (e.g. answer == prompt, or all one char)
    """
    prompt = (prompt or "").strip()
    answer = (answer or "").strip()
    if not prompt or not answer:
        return False
    combined = f"{prompt}\n\n{answer}"
    if len(combined) < min_chars:
        return False
    if _alpha_ratio(combined) < 0.3:
        return False
    if answer.strip().lower() == prompt.strip().lower():
        return False
    lowered = combined.lower()
    if any(marker in lowered for marker in _JUNK_MARKERS):
        return False
    return True


_TRANSCRIPT_JUNK_MARKERS = ("[music]", "[applause]", "[laughter]", "♪ ♪ ♪")


def passes_transcript_quality_filter(text: str, min_doc_chars: int = 300) -> bool:
    """Quality bar for video/audio transcripts -- these fail differently
    than scraped HTML articles (ASR noise, music-only stretches, stuck
    captions repeating one phrase), so this checks those patterns rather
    than the HTML junk-marker list."""
    if len(text) < min_doc_chars:
        return False
    if _alpha_ratio(text) < 0.5:
        return False
    if _top_word_repetition_ratio(text) > 0.35:
        return False
    lowered = text.lower()
    stripped = re.sub(r"\[music\]|\[applause\]|\[laughter\]|\u266a", "", lowered)
    if len(stripped) < min_doc_chars * 0.5:
        return False
    return True


CODE_ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cc", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".sh", ".sql", ".r", ".m", ".jl", ".lua", ".ml", ".hs", ".erl", ".ex",
}
CODE_SKIP_PATH_MARKERS = (
    "node_modules/", "vendor/", "third_party/", "dist/", "build/",
    ".min.js", ".min.css", "-lock.json", ".lock", "generated", ".pb.go",
)
CODE_MAX_LINE_LEN = 1000
CODE_MAX_AVG_LINE_LEN = 200


def passes_code_quality_filter(text: str, path: str, min_doc_chars: int = 500) -> bool:
    if len(text) < min_doc_chars:
        return False
    lower_path = path.lower()
    ext = os.path.splitext(lower_path)[1]
    if ext and ext not in CODE_ALLOWED_EXTENSIONS:
        return False
    if any(marker in lower_path for marker in CODE_SKIP_PATH_MARKERS):
        return False
    lines = text.split("\n")
    if any(len(l) > CODE_MAX_LINE_LEN for l in lines):
        return False
    avg_line_len = sum(len(l) for l in lines) / max(1, len(lines))
    if avg_line_len > CODE_MAX_AVG_LINE_LEN:
        return False
    return True


class ExactDedup:
    """Streaming exact-duplicate filter (sha1 digest set, not full text)."""

    def __init__(self, persist_path: Optional[str] = None):
        self._seen: set = set()
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            with open(persist_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._seen.add(bytes.fromhex(line))

    def is_duplicate(self, text: str) -> bool:
        h = hashlib.sha1(text.encode("utf-8", errors="ignore")).digest()
        if h in self._seen:
            return True
        self._seen.add(h)
        if self.persist_path:
            with open(self.persist_path, "a") as f:
                f.write(h.hex() + "\n")
        return False


class _ShingleNearDedup:
    """Fallback near-duplicate filter (no extra dependency): hashes the
    sorted set of 5-word shingles down to a fixed-size fingerprint. Two docs
    that share >= threshold fraction of shingles are treated as duplicates.
    O(n) comparisons against every prior fingerprint -- fine at the scale of
    a single run but not a substitute for real LSH at millions of docs."""

    def __init__(self, shingle_size: int = 5, threshold: float = 0.8):
        self.shingle_size = shingle_size
        self.threshold = threshold
        self._fingerprints = []

    def _shingles(self, text: str):
        words = _WORD_RE.findall(text.lower())
        n = self.shingle_size
        return {
            hashlib.md5(" ".join(words[i:i + n]).encode()).hexdigest()
            for i in range(0, max(0, len(words) - n + 1), n)
        }

    def is_near_duplicate(self, text: str) -> bool:
        shingles = self._shingles(text)
        if not shingles:
            return False
        sample = shingles if len(shingles) <= 500 else set(list(shingles)[:500])
        for fp in self._fingerprints:
            inter = len(sample & fp)
            union = len(sample | fp)
            if union and inter / union >= self.threshold:
                return True
        if len(self._fingerprints) > 5000:
            self._fingerprints.pop(0)
        self._fingerprints.append(sample)
        return False


class _MinHashLSHNearDedup:
    """Real MinHash + LSH near-dedup via `datasketch`. LSH lookup per doc
    instead of comparing against every prior fingerprint, so this scales to
    far more docs than the shingle-set fallback above. Same public
    interface (is_near_duplicate), so callers don't care which one they got."""

    def __init__(self, shingle_size: int = 5, num_perm: int = 128, threshold: float = 0.8):
        from datasketch import MinHash, MinHashLSH
        self._MinHash = MinHash
        self.shingle_size = shingle_size
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._counter = 0

    def _minhash(self, text: str):
        words = _WORD_RE.findall(text.lower())
        n = self.shingle_size
        mh = self._MinHash(num_perm=self.num_perm)
        for i in range(0, max(0, len(words) - n + 1), n):
            mh.update(" ".join(words[i:i + n]).encode())
        return mh

    def is_near_duplicate(self, text: str) -> bool:
        mh = self._minhash(text)
        if self.lsh.query(mh):
            return True
        self._counter += 1
        self.lsh.insert(f"doc-{self._counter}", mh)
        return False


def NearDedup(shingle_size: int = 5, threshold: float = 0.8):
    """Factory: returns the datasketch-backed MinHash/LSH near-dedup if
    `datasketch` is installed (recommended once a run produces more than a
    few tens of thousands of docs per category), else falls back to the
    lightweight shingle-overlap version so the pipeline still works with
    zero extra dependencies at small scale."""
    try:
        return _MinHashLSHNearDedup(shingle_size=shingle_size, threshold=threshold)
    except ImportError:
        return _ShingleNearDedup(shingle_size=shingle_size, threshold=threshold)


class RunState:
    """Persists used-queries and seen-URLs per category to a JSON file so a
    run can be killed and resumed (or run nightly via cron) without
    re-searching queries it already tried or re-extracting URLs it already
    judged -- important once a category's shards span many process runs."""

    def __init__(self, out_dir: str, category: str):
        self.path = os.path.join(out_dir, category, ".run_state.json")
        self.used_queries: list = []
        self.seen_urls: set = set()
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.used_queries = data.get("used_queries", [])
                self.seen_urls = set(data.get("seen_urls", []))
            except Exception:
                pass  # corrupt state file -- start fresh rather than crash

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Cap what's persisted so this file doesn't grow unboundedly across
        # many resumed runs -- keep the most recent queries/URLs, which is
        # what the planner's "avoid repeating" prompt and the dedup check
        # actually need.
        with open(self.path, "w") as f:
            json.dump({
                "used_queries": self.used_queries[-2000:],
                "seen_urls": list(self.seen_urls)[-50000:],
            }, f)


class ShardWriter:
    """Writes JSONL records to size-capped shard files, identical layout to
    build_dataset.py's ShardWriter so downstream pack_*.py scripts need zero
    changes to consume agent-produced shards."""

    def __init__(self, out_dir: str, category: str, max_shard_bytes: int = SHARD_MAX_BYTES):
        self.dir = os.path.join(out_dir, category)
        os.makedirs(self.dir, exist_ok=True)
        self.category = category
        self.max_shard_bytes = max_shard_bytes
        # Resume from the highest existing shard index instead of always
        # starting at 0 (which would overwrite previous agent runs).
        existing = [f for f in os.listdir(self.dir) if f.startswith(f"{category}_") and f.endswith(".jsonl")]
        self.shard_idx = max((int(f[len(category) + 1: -6]) for f in existing), default=-1) + 1
        self.bytes_in_shard = 0
        self.total_bytes = 0
        self.total_docs = 0
        self._fh = self._open_new_shard()

    def _open_new_shard(self):
        path = os.path.join(self.dir, f"{self.category}_{self.shard_idx:05d}.jsonl")
        return open(path, "w", encoding="utf-8")

    def write(self, record: dict):
        line = json.dumps(record, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        if self.bytes_in_shard + line_bytes > self.max_shard_bytes and self.bytes_in_shard > 0:
            self._fh.close()
            self.shard_idx += 1
            self.bytes_in_shard = 0
            self._fh = self._open_new_shard()

        self._fh.write(line)
        self.bytes_in_shard += line_bytes
        self.total_bytes += line_bytes
        self.total_docs += 1

    def close(self):
        self._fh.close()
