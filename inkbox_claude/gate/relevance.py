"""Shared lexical relevance: a cheap word-overlap score used to decide which
ledger events and vault note sections are worth attaching to a run's context,
instead of attaching everything whole. This is token overlap, not a model
call, so it costs nothing measurable on top of a request already in flight."""
from __future__ import annotations

import re
from typing import FrozenSet, Set

_WORD_RE = re.compile(r"[a-z0-9]+")

# Ordinary grammatical words, dropped so matches reflect shared topic words
# (names, projects, nouns) rather than words every message has in common.
STOPWORDS: FrozenSet[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "and", "or", "with", "at", "by", "from",
    "what", "whats", "when", "where", "who", "whom", "which", "how", "why",
    "does", "do", "did", "doing", "it", "its", "this", "that", "these",
    "those", "my", "me", "i", "you", "your", "he", "she", "they", "we", "us",
    "about", "any", "all", "can", "could", "would", "should", "will",
    "has", "have", "had", "not", "no", "yes", "then", "than", "but", "if",
    "there", "here", "into", "out", "up", "down", "so", "just",
})


def tokens(text: str) -> Set[str]:
    """Lowercase alphanumeric words of 3+ characters, stopwords dropped."""
    if not text:
        return set()
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3 and w not in STOPWORDS}


def overlap_score(candidate: str, query: FrozenSet[str]) -> int:
    """How many of the query's words this candidate text shares."""
    if not query or not candidate:
        return 0
    return len(tokens(candidate) & query)
