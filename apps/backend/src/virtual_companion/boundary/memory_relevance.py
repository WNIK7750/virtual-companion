"""Shared lexical relevance baseline; eligibility remains the store's authority."""

import re

_FUNCTION_CHARS = frozenset("的了是我你他她它们吗呢啊吧呀哦嗯在和也都就着得地个这那么")
MAX_RECALL_QUERY_CHARS = 256
MAX_RECALL_QUERY_BYTES = 768


def validate_recall_query(value):
    """A transient search expression, never fact content or mutation authority."""
    if value is None:
        return None
    if (not isinstance(value, str) or not value.strip()
            or len(value) > MAX_RECALL_QUERY_CHARS
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("Bounded plain-text memory query required")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError("Unicode memory query required") from error
    if size > MAX_RECALL_QUERY_BYTES:
        raise ValueError("Memory query exceeds UTF-8 bound")
    return value.strip()


def _terms(value):
    compact = "".join(value.lower().split())
    chars = {char for char in compact
             if "\u4e00" <= char <= "\u9fff" and char not in _FUNCTION_CHARS}
    bigrams = {compact[index:index + 2] for index in range(max(0, len(compact) - 1))
               if any(char.isalnum() and char not in _FUNCTION_CHARS
                      for char in compact[index:index + 2])}
    words = set(re.findall(r"[A-Za-z0-9_]+", value.lower()))
    return chars, bigrams, words


def lexical_relevance(query, *, include_unmatched=False, recall_query=None):
    """Keep single-character topics, excluding overlap consisting only of grammar.

    Literal overlap is a retrieval cue, not a semantic or source-validity proof.
    Word and phrase matches still rank ahead of a shared Chinese content character.
    Explicit predicate phrases rank below content matches, but above unmatched
    candidates. Predicate matching requires two Chinese content characters or a
    whole word; a shared character or grammar alone is insufficient.
    After a caller explicitly requests memory, unmatched eligible records may be
    supplied as lower-priority candidates for the existing answer model to judge.
    A bounded turn-local expression replaces the default search terms only. It cannot
    select record IDs or bypass store filtering and the caller's context limit.
    """
    recall_query = validate_recall_query(recall_query)
    if recall_query is not None:
        query = recall_query
    query_chars, query_bigrams, query_words = _terms(query)
    query_predicate_phrases = {
        phrase for phrase in query_bigrams
        if all("\u4e00" <= char <= "\u9fff" and char not in _FUNCTION_CHARS for char in phrase)
    }

    def score(value):
        chars, bigrams, words = _terms(value.get("content", ""))
        relevance = (
            2 * len(query_bigrams & bigrams)
            + len(query_chars & chars)
            + 2 * len(query_words & words)
        )
        if relevance:
            return relevance
        predicate = (value.get("fact_key") or {}).get("predicate", "")
        _, predicate_bigrams, predicate_words = _terms(predicate)
        if query_predicate_phrases & predicate_bigrams or query_words & predicate_words:
            return 0.75
        return 0.5 if include_unmatched else 0

    return score
