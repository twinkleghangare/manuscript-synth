"""Corpus loading and passage sampling."""
from pathlib import Path


def load_corpus(path):
    p = Path(path)
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def sample_passage(rng, corpus, n_words):
    """Concatenate random corpus lines until ~n_words words are collected."""
    words = []
    while len(words) < n_words:
        words += corpus[int(rng.integers(len(corpus)))].split()
    return " ".join(words[:n_words])
