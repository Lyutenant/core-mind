from __future__ import annotations

import pytest

from coremind import WakeWordError
from coremind.wake_word.dummy import DummyWakeWordDetector


# ---------------------------------------------------------------------------
# DummyWakeWordDetector
# ---------------------------------------------------------------------------

def test_dummy_trigger_prompt():
    det = DummyWakeWordDetector()
    assert "Enter" in det.trigger_prompt


def test_dummy_listen_calls_input(mocker):
    mocker.patch("builtins.input", return_value="")
    det = DummyWakeWordDetector()
    det.listen_until_wake_word()  # should not raise


# ---------------------------------------------------------------------------
# OpenWakeWordDetector
# ---------------------------------------------------------------------------

def test_openwakeword_raises_if_not_installed(mocker):
    mocker.patch.dict("sys.modules", {"openwakeword": None, "openwakeword.model": None})
    import importlib
    import coremind.wake_word.openwakeword_engine as mod
    importlib.reload(mod)

    with pytest.raises(WakeWordError, match="openwakeword is not installed"):
        mod.OpenWakeWordDetector(model="hey_jarvis_v0.1")


def test_openwakeword_trigger_prompt_contains_model(mocker):
    mock_model_cls = mocker.MagicMock()
    mock_model_cls.return_value = mocker.MagicMock()
    mocker.patch.dict(
        "sys.modules",
        {
            "openwakeword": mocker.MagicMock(),
            "openwakeword.model": mocker.MagicMock(Model=mock_model_cls),
        },
    )
    import importlib
    import coremind.wake_word.openwakeword_engine as mod
    importlib.reload(mod)

    det = mod.OpenWakeWordDetector(model="hey_test_v1")
    assert "hey_test_v1" in det.trigger_prompt


def test_openwakeword_raises_on_bad_model(mocker):
    mock_model_cls = mocker.MagicMock()
    mock_model_cls.side_effect = Exception("model not found")
    mocker.patch.dict(
        "sys.modules",
        {
            "openwakeword": mocker.MagicMock(),
            "openwakeword.model": mocker.MagicMock(Model=mock_model_cls),
        },
    )
    import importlib
    import coremind.wake_word.openwakeword_engine as mod
    importlib.reload(mod)

    with pytest.raises(WakeWordError, match="Failed to load"):
        mod.OpenWakeWordDetector(model="nonexistent")
