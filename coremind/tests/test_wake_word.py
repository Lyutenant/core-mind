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


# ---------------------------------------------------------------------------
# Silero VAD pre-gate (vad_threshold)
# ---------------------------------------------------------------------------

def _load_engine_with_mock_model(mocker, model_cls, vad_cls=None):
    modules = {
        "openwakeword": mocker.MagicMock(),
        "openwakeword.model": mocker.MagicMock(Model=model_cls),
    }
    if vad_cls is not None:
        modules["openwakeword.vad"] = mocker.MagicMock(VAD=vad_cls)
    mocker.patch.dict("sys.modules", modules)
    import importlib
    import coremind.wake_word.openwakeword_engine as mod
    importlib.reload(mod)
    return mod


def test_vad_threshold_default_off_builds_as_before(mocker):
    mock_model_cls = mocker.MagicMock()
    mod = _load_engine_with_mock_model(mocker, mock_model_cls)

    det = mod.OpenWakeWordDetector(model="hey_test_v1")  # vad_threshold defaults to 0.0
    # Default must not touch the Silero VAD at all (no new startup dependency).
    assert "vad_threshold" not in mock_model_cls.call_args.kwargs
    assert det._vad_threshold == 0.0
    assert det._vad_loaded is False


def test_vad_threshold_loads_when_enabled(mocker):
    mock_model_cls = mocker.MagicMock()
    mod = _load_engine_with_mock_model(mocker, mock_model_cls)

    det = mod.OpenWakeWordDetector(model="hey_test_v1", vad_threshold=0.3)
    # Gate enabled → Model built with the configured vad_threshold (Silero VAD loads).
    assert mock_model_cls.call_args.kwargs["vad_threshold"] == 0.3
    assert det._vad_loaded is True
    assert det._vad_threshold == 0.3


def test_vad_threshold_older_openwakeword(mocker):
    # Model() raises TypeError when passed vad_threshold (older openwakeword),
    # but succeeds without it — detector must degrade to no-VAD, not raise.
    def _model(*args, **kwargs):
        if "vad_threshold" in kwargs:
            raise TypeError("unexpected keyword argument 'vad_threshold'")
        return mocker.MagicMock()

    mock_model_cls = mocker.MagicMock(side_effect=_model)
    mod = _load_engine_with_mock_model(mocker, mock_model_cls)

    det = mod.OpenWakeWordDetector(model="hey_test_v1", vad_threshold=0.3)  # must not raise
    assert det._vad_loaded is False
    assert det._vad_threshold == 0.0


def test_vad_load_failure_degrades_gracefully(mocker):
    # Silero VAD assets missing/broken: Model() raises a generic error when the gate is
    # requested but succeeds without it. Startup must NOT fail just because the gate is on.
    def _model(*args, **kwargs):
        if "vad_threshold" in kwargs:
            raise RuntimeError("could not load Silero VAD model")
        return mocker.MagicMock()

    mock_model_cls = mocker.MagicMock(side_effect=_model)
    mod = _load_engine_with_mock_model(mocker, mock_model_cls)

    det = mod.OpenWakeWordDetector(model="hey_test_v1", vad_threshold=0.5)  # must not raise
    assert det._vad_loaded is False
    assert det._vad_threshold == 0.0


def test_set_vad_threshold_lazy_loads_from_off(mocker):
    # A node that started at the default 0.0 must be able to ENABLE the gate at runtime
    # (Hub override / live slider) — Silero is loaded on demand, no restart.
    fake_oww = mocker.MagicMock()
    fake_oww.vad = None  # no VAD loaded yet
    mock_model_cls = mocker.MagicMock(return_value=fake_oww)
    vad_cls = mocker.MagicMock()
    mod = _load_engine_with_mock_model(mocker, mock_model_cls, vad_cls=vad_cls)

    det = mod.OpenWakeWordDetector(model="hey_test_v1")  # starts off
    assert det._vad_loaded is False

    assert det.set_vad_threshold(0.4) is True
    assert det._vad_loaded is True
    vad_cls.assert_called_once()              # Silero loaded on demand
    assert det._vad_threshold == 0.4
    assert fake_oww.vad_threshold == 0.4


def test_set_vad_threshold_runtime_load_failure_stays_off(mocker):
    # If Silero can't be loaded at runtime, the gate stays off and we don't crash.
    fake_oww = mocker.MagicMock()
    fake_oww.vad = None
    mock_model_cls = mocker.MagicMock(return_value=fake_oww)
    vad_cls = mocker.MagicMock(side_effect=RuntimeError("no silero assets"))
    mod = _load_engine_with_mock_model(mocker, mock_model_cls, vad_cls=vad_cls)

    det = mod.OpenWakeWordDetector(model="hey_test_v1")
    assert det.set_vad_threshold(0.4) is False
    assert det._vad_loaded is False
    assert det._vad_threshold == 0.0
    assert fake_oww.vad_threshold == 0.0


def test_set_vad_threshold_refused_on_unsupported_build(mocker):
    # openwakeword build that rejects vad_threshold at construction: the live slider must
    # NOT report a phantom-active gate, because predict() would ignore vad_threshold.
    def _model(*args, **kwargs):
        if "vad_threshold" in kwargs:
            raise TypeError("unexpected keyword argument 'vad_threshold'")
        return mocker.MagicMock()

    mock_model_cls = mocker.MagicMock(side_effect=_model)
    vad_cls = mocker.MagicMock()
    mod = _load_engine_with_mock_model(mocker, mock_model_cls, vad_cls=vad_cls)

    det = mod.OpenWakeWordDetector(model="hey_test_v1", vad_threshold=0.3)
    assert det._vad_supported is False

    assert det.set_vad_threshold(0.4) is False  # refused
    assert det._vad_threshold == 0.0
    assert det._vad_loaded is False
    vad_cls.assert_not_called()  # never even tried to load Silero


def test_set_vad_threshold_disable_keeps_loaded(mocker):
    # Disabling the gate (set 0.0) must not unload Silero, so it can be re-enabled live.
    mock_model_cls = mocker.MagicMock()
    mod = _load_engine_with_mock_model(mocker, mock_model_cls)

    det = mod.OpenWakeWordDetector(model="hey_test_v1", vad_threshold=0.5)
    assert det._vad_loaded is True

    det.set_vad_threshold(0.0)
    assert det._vad_threshold == 0.0
    assert det._oww.vad_threshold == 0.0
    assert det._vad_loaded is True  # still loaded — re-enabling stays live
