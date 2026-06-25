from __future__ import annotations

from typing import Sequence

# Characters stripped from the edges of a token before comparison, so a trailing
# "over." / "over!" / "(over)" still matches the configured word "over".
_PUNCT = ".,!?;:\"')(][}{-–—…"


def _normalize_token(token: str) -> str:
    """Lowercase a single token and strip surrounding punctuation."""
    return token.strip(_PUNCT).lower()


def _phrase_words(phrase: str) -> list[str]:
    """Normalize a configured terminator phrase into a list of comparable tokens."""
    return [_normalize_token(w) for w in phrase.split() if _normalize_token(w)]


def match_and_strip_terminator(transcript: str, words: Sequence[str]) -> tuple[bool, str]:
    """Check whether ``transcript`` ends with one of the configured terminator phrases.

    Used by the wake-confirmation gate: the first utterance after the wake word must end
    with a configured word (e.g. "over") or it is discarded as a false wake.

    Matching is case-insensitive and ignores surrounding punctuation. A phrase matches when
    the transcript's trailing tokens equal the phrase's tokens; the LONGEST matching phrase
    wins (so "go ahead" is preferred over a bare "ahead"). On a match the matched trailing
    tokens are removed from the *original* transcript (the remaining text keeps its casing),
    with trailing whitespace/punctuation trimmed.

    Returns ``(matched, stripped)``. When ``words`` is empty/falsy returns
    ``(False, transcript)``. An utterance that is *only* the terminator yields
    ``(True, "")`` — the caller treats an empty stripped result as a discard.
    """
    if not words:
        return False, transcript

    original_tokens = transcript.split()
    if not original_tokens:
        return False, transcript
    norm_tokens = [_normalize_token(t) for t in original_tokens]

    best_len = 0
    for phrase in words:
        pw = _phrase_words(phrase)
        n = len(pw)
        if n == 0 or n > len(norm_tokens):
            continue
        if norm_tokens[-n:] == pw and n > best_len:
            best_len = n

    if best_len == 0:
        return False, transcript

    stripped = " ".join(original_tokens[:-best_len]).rstrip(_PUNCT + " ")
    return True, stripped
