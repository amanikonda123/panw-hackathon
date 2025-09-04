# src/services/nlp.py
from __future__ import annotations

import re
from typing import List, Tuple, Optional, Dict
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# ---- Sentiment (VADER) ----
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    class SentimentIntensityAnalyzer:  # neutral fallback
        def polarity_scores(self, _): return {"compound": 0.0}

_ANALYZER: Optional[SentimentIntensityAnalyzer] = None
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_MENTION_RE = re.compile(r"[@#]\w+", re.U)

def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER

def _normalize_for_sentiment(text: str) -> str:
    if not isinstance(text, str): return ""
    s = text.strip()
    s = _URL_RE.sub("", s)
    s = _MENTION_RE.sub(lambda m: m.group(0)[1:], s)  # drop @/# symbol, keep word
    return s

def sentiment_label_and_score(text: str) -> tuple[str, float]:
    a = _get_analyzer()
    s = _normalize_for_sentiment(text)
    compound = float(a.polarity_scores(s)["compound"])
    if compound >= 0.05: label = "positive"
    elif compound <= -0.05: label = "negative"
    else: label = "neutral"
    return label, compound


# ---- Keywording (TF-IDF first, with graceful fallback) ----

# Words we don't want as themes even if frequent
FILLER_STOP = {
    "feeling","feel","pretty","really","very","so","just","kind","sort","right",
    "now","today","tonight","morning","afternoon","evening","bit","little","lot",
    "things","thing","stuff","okay","ok","good","bad","nice","great","awesome",
    "cool","fine","yay","meh", "time", "today", "tomorrow", "yesterday", "actually", "really",
    "thing", "things", "stuff", "pay"
}

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"

def _make_stopwords() -> set:
    sw = set(ENGLISH_STOP_WORDS)
    sw.update(FILLER_STOP)
    return sw

def _prep_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.strip()
    text = _URL_RE.sub(" ", text)
    text = text.replace("\n", " ")
    return text

def _tfidf_pairs(
    docs: List[str],
    for_doc_index: int,
    stop_set: set,
    ngram_range=(1,3),
) -> List[Tuple[str, float]]:
    """Return all (term, score) for the target doc, sorted by score desc."""
    if not docs or for_doc_index >= len(docs):
        return []
    n_docs = len(docs)
    max_df = 1.0 if n_docs <= 3 else 0.95  # avoid small-corpus crash
    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        stop_words=list(stop_set),   # <-- cast to list (IMPORTANT)
        ngram_range=ngram_range,
        min_df=1,
        max_df=max_df,
        sublinear_tf=True,          # (optional) smoother tf
        norm="l2",
    )
    X = vec.fit_transform(docs)  # (n_docs, vocab)
    if X.shape[1] == 0:
        return []
    row = X[for_doc_index].toarray().ravel()
    feats = vec.get_feature_names_out()
    idx = np.argsort(-row)  # descending by score
    return [(feats[i], float(row[i])) for i in idx if row[i] > 0]

def _dedupe_by_containment_ordered(pairs: List[Tuple[str,float]], k: int) -> List[str]:
    """
    Keep highest-scoring terms first; drop a term if it is a substring of any kept term.
    Returns top-k strings in the same (score) order.
    """
    kept: List[str] = []
    for term, _ in pairs:
        t = term.strip().lower()
        if not t: continue
        if any(t in big for big in kept if big != t):
            continue
        kept.append(t)
        if len(kept) >= k:
            break
    return kept

def _simple_fallback(current_text: str, stop_set: set, k: int) -> List[str]:
    """
    If TF-IDF yields nothing (super short text, all stopwords), extract from current text only.
    Prefer bigrams by frequency, then backfill with unigrams.
    """
    tokens = re.findall(TOKEN_PATTERN, (current_text or "").lower())
    tokens = [t for t in tokens if t not in stop_set]
    if not tokens: return []
    bigrams = [" ".join(p) for p in zip(tokens, tokens[1:])]
    c_bi = Counter(bigrams)
    c_uni = Counter(tokens)

    tmp: List[str] = []
    for p,_ in c_bi.most_common(k*2):
        tmp.append(p)
    for u,_ in c_uni.most_common(k*2):
        if not any(u in m for m in tmp):
            tmp.append(u)

    out, seen = [], set()
    for t in tmp:
        if t in seen: continue
        seen.add(t)
        out.append(t)
        if len(out) >= k: break
    return out

def extract_keyphrases_sorted_by_tfidf(
    current_text: str,
    history_texts: List[str],
    k: int = 8,
) -> List[str]:
    """
    Returns top-k phrases **sorted by TF-IDF relevance** for the current doc.
    - Uses n-grams (1–3)
    - Dedupes by containment (keeps higher-scoring multiwords over their unigrams)
    - Falls back to simple frequency-based extraction if TF-IDF has no vocab
    """
    stop_set = _make_stopwords()
    corpus = [ _prep_text(t or "") for t in (list(history_texts) + [current_text or ""]) ]
    target = len(corpus) - 1

    pairs = _tfidf_pairs(corpus, target, stop_set, ngram_range=(1,3))
    if pairs:
        # Optional tie-breaker for equal scores: prefer more words & longer strings
        # (We maintain primary order by TF-IDF via stable sort)
        def _key(p):  # not used directly; _dedupe preserves order
            term, score = p
            return (-score, -(term.count(" ")+1), -len(term))
        # Deduplicate while preserving the TF-IDF order
        ordered = _dedupe_by_containment_ordered(pairs, k)
        return ordered

    # Fallback if TF-IDF produced no terms
    return _simple_fallback(current_text, stop_set, k)


# ---- Public API (used by UI) ----

def analyze_entry(entry_text: str, history_texts: List[str]) -> Dict[str, object]:
    """
    Analyze a single entry:
      - VADER sentiment (label + compound)
      - TF-IDF keyphrases for this entry, sorted by relevance (length k=6)
    """
    if not isinstance(entry_text, str): entry_text = ""
    s_label, s_comp = sentiment_label_and_score(entry_text)
    keywords = extract_keyphrases_sorted_by_tfidf(entry_text, history_texts, k=6)
    return {
        "sentiment_label": s_label,
        "sentiment_score": float(s_comp),
        "keywords": keywords,
    }


def personalized_sentiment(
    current_compound: float,
    history_compounds: List[float],
    window: int = 30,
    z_pos: float = 0.5,
    z_neg: float = -0.5,
) -> dict:
    """
    Compare today's compound to the user's own recent baseline (last `window` entries).
    Returns {'z', 'label', 'n_history', 'mean', 'std'}.
    """
    hist = [float(x) for x in history_compounds][-window:]
    n = len(hist)
    if n == 0:
        return {"z": 0.0, "label": "near baseline", "n_history": 0, "mean": 0.0, "std": 0.0}

    mean = float(np.mean(hist))
    std = float(np.std(hist))
    std = max(std, 0.05)  # guard against tiny variance

    z = float((float(current_compound) - mean) / std)
    if z >= z_pos: label = "above baseline"
    elif z <= z_neg: label = "below baseline"
    else: label = "near baseline"

    return {"z": z, "label": label, "n_history": n, "mean": mean, "std": std}
