from __future__ import annotations

from coremind.text_match import match_and_strip_terminator

WORDS = ["over", "go ahead", "confirm"]


def test_single_word_match_and_strip():
    assert match_and_strip_terminator("what time is it over", WORDS) == (
        True,
        "what time is it",
    )


def test_trailing_punctuation_is_ignored():
    assert match_and_strip_terminator("what's the weather over.", WORDS) == (
        True,
        "what's the weather",
    )


def test_case_insensitive():
    assert match_and_strip_terminator("Play some music OVER", WORDS) == (
        True,
        "Play some music",
    )


def test_two_word_phrase_match():
    assert match_and_strip_terminator("play music go ahead", WORDS) == (
        True,
        "play music",
    )


def test_longest_phrase_wins():
    # "ahead" alone is not a terminator; "go ahead" is — the two-word phrase must be stripped.
    assert match_and_strip_terminator("start the timer go ahead", ["ahead", "go ahead"]) == (
        True,
        "start the timer",
    )


def test_no_terminator_does_not_match():
    assert match_and_strip_terminator("the meeting is finished", WORDS) == (
        False,
        "the meeting is finished",
    )


def test_terminator_only_strips_to_empty():
    assert match_and_strip_terminator("over", WORDS) == (True, "")


def test_terminator_must_be_at_the_end():
    # The word appears but not as the final token — not a valid confirmation.
    assert match_and_strip_terminator("over the rainbow please", WORDS) == (
        False,
        "over the rainbow please",
    )


def test_empty_words_disables_gate():
    assert match_and_strip_terminator("anything at all", []) == (False, "anything at all")


def test_empty_transcript():
    assert match_and_strip_terminator("", WORDS) == (False, "")
