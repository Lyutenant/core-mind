from __future__ import annotations

import json

import pytest

from coremind import device_cache
from coremind.audio_input import devices as dev
from coremind.audio_input.devices import (
    AudioDevice,
    auto_select_input_name,
    auto_select_output_name,
    coerce_device,
    resolve_input_device,
    resolve_output_device,
)
from coremind.audio_input.recorder import Recorder
from coremind.audio_output.player import Player


def _mic(name: str) -> AudioDevice:
    return AudioDevice(index=0, name=name, max_input_channels=1, max_output_channels=0, default_sample_rate=48000)


def _spk(name: str) -> AudioDevice:
    return AudioDevice(index=1, name=name, max_input_channels=0, max_output_channels=2, default_sample_rate=48000)


# ---------------------------------------------------------------------------
# coerce_device
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("1", 1),
    (" 2 ", 2),
    ("USB Audio", "USB Audio"),
    (3, 3),
    (None, None),
])
def test_coerce_device(value, expected):
    assert coerce_device(value) == expected


# ---------------------------------------------------------------------------
# auto-selection heuristics (cache disabled unless stated)
# ---------------------------------------------------------------------------

@pytest.fixture
def no_cache(mocker):
    mocker.patch.object(device_cache, "get", return_value=None)
    mocker.patch.object(device_cache, "remember")


def test_auto_input_prefers_usb_over_builtin(mocker, no_cache):
    mocker.patch.object(dev, "list_input_devices", return_value=[_mic("Built-in Mic"), _mic("USB PnP Sound Device")])
    mocker.patch.object(dev, "get_default_input_device", return_value=None)
    assert auto_select_input_name() == "USB PnP Sound Device"


def test_auto_input_deprioritizes_webcam_mic(mocker, no_cache):
    # A UVC webcam exposes a mic too; the standalone USB mic should win.
    mocker.patch.object(dev, "list_input_devices", return_value=[_mic("USB Camera: Webcam"), _mic("USB Audio CODEC")])
    mocker.patch.object(dev, "get_default_input_device", return_value=None)
    assert auto_select_input_name() == "USB Audio CODEC"


def test_auto_input_falls_back_to_default_then_first(mocker, no_cache):
    mocker.patch.object(dev, "list_input_devices", return_value=[_mic("Analog In"), _mic("HDMI In")])
    mocker.patch.object(dev, "get_default_input_device", return_value=_mic("HDMI In"))
    assert auto_select_input_name() == "HDMI In"

    mocker.patch.object(dev, "get_default_input_device", return_value=None)
    assert auto_select_input_name() == "Analog In"


def test_auto_input_none_when_no_devices(mocker, no_cache):
    mocker.patch.object(dev, "list_input_devices", return_value=[])
    mocker.patch.object(dev, "get_default_input_device", return_value=None)
    assert auto_select_input_name() is None


def test_auto_output_prefers_usb(mocker, no_cache):
    mocker.patch.object(dev, "list_output_devices", return_value=[_spk("HDMI Out"), _spk("USB Speaker")])
    mocker.patch.object(dev, "get_default_output_device", return_value=None)
    assert auto_select_output_name() == "USB Speaker"


def test_auto_output_skips_mic_array(mocker, no_cache):
    # A USB mic array (reSpeaker) also exposes a playback endpoint that comes
    # first in enumeration; the real USB speaker should win, not the mic array.
    mocker.patch.object(dev, "list_output_devices", return_value=[
        _spk("reSpeaker XVF3800 4-Mic Array: USB Audio (hw:2,0)"),
        _spk("USB2.0 Device: Audio (hw:3,0)"),
    ])
    mocker.patch.object(dev, "get_default_output_device", return_value=None)
    assert auto_select_output_name() == "USB2.0 Device: Audio (hw:3,0)"


def test_auto_output_uses_mic_array_only_when_sole_usb(mocker, no_cache):
    # If the mic array is the *only* USB output, fall back to it rather than None.
    mocker.patch.object(dev, "list_output_devices", return_value=[
        _spk("reSpeaker XVF3800 4-Mic Array: USB Audio (hw:2,0)"),
    ])
    mocker.patch.object(dev, "get_default_output_device", return_value=None)
    assert auto_select_output_name() == "reSpeaker XVF3800 4-Mic Array: USB Audio (hw:2,0)"


def test_auto_output_prefers_default_over_rejected_usb(mocker, no_cache):
    # The only USB output is a mic array, but a normal default speaker exists →
    # the default should win, not the silent mic-array endpoint.
    mocker.patch.object(dev, "list_output_devices", return_value=[
        _spk("reSpeaker XVF3800 4-Mic Array: USB Audio (hw:2,0)"),
        _spk("Built-in Output"),
    ])
    mocker.patch.object(dev, "get_default_output_device",
                        return_value=_spk("Built-in Output"))
    assert auto_select_output_name() == "Built-in Output"


def test_auto_output_self_heals_stale_mic_array_cache(mocker):
    # A cache written before mic-array avoidance existed points at the reSpeaker;
    # it should be ignored in favour of the real speaker (no manual cache wipe).
    mocker.patch.object(device_cache, "get", return_value="reSpeaker XVF3800 4-Mic Array")
    mocker.patch.object(device_cache, "remember")
    mocker.patch.object(dev, "list_output_devices", return_value=[
        _spk("reSpeaker XVF3800 4-Mic Array: USB Audio (hw:2,0)"),
        _spk("USB2.0 Device: Audio (hw:3,0)"),
    ])
    mocker.patch.object(dev, "get_default_output_device", return_value=None)
    assert auto_select_output_name() == "USB2.0 Device: Audio (hw:3,0)"


def test_auto_input_uses_cache_when_present(mocker):
    mocker.patch.object(device_cache, "get", return_value="Scarlett")
    mocker.patch.object(dev, "list_input_devices", return_value=[_mic("Focusrite Scarlett 2i2"), _mic("USB PnP Sound Device")])
    mocker.patch.object(dev, "get_default_input_device", return_value=None)
    # Cached substring still matches a present device → reuse it (don't re-pick USB).
    assert auto_select_input_name() == "Scarlett"


def test_auto_input_ignores_stale_cache(mocker):
    mocker.patch.object(device_cache, "get", return_value="Gone Device")
    mocker.patch.object(dev, "list_input_devices", return_value=[_mic("USB PnP Sound Device")])
    mocker.patch.object(dev, "get_default_input_device", return_value=None)
    assert auto_select_input_name() == "USB PnP Sound Device"


# ---------------------------------------------------------------------------
# resolve_* (explicit pin vs. auto)
# ---------------------------------------------------------------------------

def test_resolve_input_explicit_index_passthrough(mocker):
    auto = mocker.patch.object(dev, "auto_select_input_name")
    assert resolve_input_device(1) == 1
    assert resolve_input_device("2") == 2
    assert resolve_input_device("USB Mic") == "USB Mic"
    auto.assert_not_called()


def test_resolve_input_auto_selects_and_persists(mocker):
    mocker.patch.object(dev, "auto_select_input_name", return_value="USB Mic")
    remember = mocker.patch.object(device_cache, "remember")
    for configured in (None, "auto"):
        assert resolve_input_device(configured) == "USB Mic"
    remember.assert_called_with(device_cache.INPUT_KEY, "USB Mic")


def test_resolve_output_auto_selects_and_persists(mocker):
    mocker.patch.object(dev, "auto_select_output_name", return_value="USB Speaker")
    remember = mocker.patch.object(device_cache, "remember")
    assert resolve_output_device(None) == "USB Speaker"
    remember.assert_called_with(device_cache.OUTPUT_KEY, "USB Speaker")


# ---------------------------------------------------------------------------
# Recorder/Player accept a string device name
# ---------------------------------------------------------------------------

def test_recorder_stores_string_device():
    assert Recorder(device="USB Audio").device == "USB Audio"


def test_player_stores_string_device():
    assert Player(device="USB Speaker").device == "USB Speaker"


# ---------------------------------------------------------------------------
# device_cache persistence
# ---------------------------------------------------------------------------

def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(device_cache, "_CACHE_PATH", tmp_path / "device-cache.json")
    assert device_cache.get(device_cache.INPUT_KEY) is None
    device_cache.remember(device_cache.INPUT_KEY, "USB Mic")
    device_cache.remember(device_cache.OUTPUT_KEY, "USB Speaker")
    assert device_cache.get(device_cache.INPUT_KEY) == "USB Mic"
    assert device_cache.get(device_cache.OUTPUT_KEY) == "USB Speaker"
    # Both keys persisted to the same file.
    data = json.loads((tmp_path / "device-cache.json").read_text())
    assert data == {"input_device_name": "USB Mic", "output_device_name": "USB Speaker"}


def test_cache_tolerates_corrupt_file(tmp_path, monkeypatch):
    path = tmp_path / "device-cache.json"
    path.write_text("{not valid json")
    monkeypatch.setattr(device_cache, "_CACHE_PATH", path)
    assert device_cache.load() == {}
    assert device_cache.get(device_cache.INPUT_KEY) is None


# ---------------------------------------------------------------------------
# AudioConfig device coercion (numeric strings → int index)
# ---------------------------------------------------------------------------

def test_audio_config_coerces_numeric_string_device():
    from coremind.config.settings import AudioConfig

    # Env-var-style numeric strings become int indices...
    cfg = AudioConfig(input_device="1", output_device="2")
    assert cfg.input_device == 1
    assert cfg.output_device == 2

    # ...but a real name substring is preserved, and None stays None (auto).
    named = AudioConfig(input_device="USB Audio CODEC", output_device=None)
    assert named.input_device == "USB Audio CODEC"
    assert named.output_device is None
