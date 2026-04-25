from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from coremind import AudioInputError, AudioOutputError
from coremind.audio_input.devices import AudioDevice, list_input_devices, list_output_devices
from coremind.audio_input.recorder import Recorder
from coremind.audio_output.player import Player


# ---------------------------------------------------------------------------
# AudioDevice dataclass
# ---------------------------------------------------------------------------

def test_audio_device_is_input():
    d = AudioDevice(index=0, name="Mic", max_input_channels=2, max_output_channels=0, default_sample_rate=44100)
    assert d.is_input()
    assert not d.is_output()


def test_audio_device_is_output():
    d = AudioDevice(index=1, name="Speaker", max_input_channels=0, max_output_channels=2, default_sample_rate=48000)
    assert d.is_output()
    assert not d.is_input()


def test_audio_device_both():
    d = AudioDevice(index=2, name="Headset", max_input_channels=1, max_output_channels=2, default_sample_rate=44100)
    assert d.is_input()
    assert d.is_output()


# ---------------------------------------------------------------------------
# Device listing (mocked sounddevice)
# ---------------------------------------------------------------------------

_FAKE_DEVICES = [
    {"name": "Built-in Mic", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100.0},
    {"name": "Built-in Output", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0},
    {"name": "USB Headset", "max_input_channels": 1, "max_output_channels": 2, "default_samplerate": 48000.0},
]


def test_list_input_devices(mocker):
    mocker.patch("sounddevice.query_devices", return_value=_FAKE_DEVICES)
    devices = list_input_devices()
    assert len(devices) == 2
    assert all(d.is_input() for d in devices)
    assert devices[0].name == "Built-in Mic"
    assert devices[1].name == "USB Headset"


def test_list_output_devices(mocker):
    mocker.patch("sounddevice.query_devices", return_value=_FAKE_DEVICES)
    devices = list_output_devices()
    assert len(devices) == 2
    assert all(d.is_output() for d in devices)
    assert devices[0].name == "Built-in Output"
    assert devices[1].name == "USB Headset"


def test_list_devices_portaudio_error(mocker):
    import sounddevice as sd
    mocker.patch("sounddevice.query_devices", side_effect=sd.PortAudioError("no audio"))
    with pytest.raises(AudioInputError, match="Failed to query"):
        list_input_devices()


# ---------------------------------------------------------------------------
# Recorder (mocked sounddevice + soundfile)
# ---------------------------------------------------------------------------

def test_recorder_saves_wav(mocker, tmp_path):
    fake_audio = np.zeros((16000, 1), dtype="float32")
    mocker.patch("sounddevice.rec", return_value=fake_audio)
    mocker.patch("sounddevice.wait")

    out = tmp_path / "out.wav"
    recorder = Recorder(sample_rate=16000, channels=1)
    result = recorder.record(seconds=1.0, output_path=str(out))

    assert result == out
    assert out.exists()
    data, sr = sf.read(str(out))
    assert sr == 16000


def test_recorder_raises_on_portaudio_error(mocker):
    import sounddevice as sd
    mocker.patch("sounddevice.rec", side_effect=sd.PortAudioError("no mic"))
    mocker.patch("sounddevice.wait")

    recorder = Recorder()
    with pytest.raises(AudioInputError, match="Recording failed"):
        recorder.record(seconds=1.0, output_path="/tmp/test.wav")


# ---------------------------------------------------------------------------
# Player (mocked sounddevice)
# ---------------------------------------------------------------------------

def test_player_plays_wav(mocker, tmp_path):
    wav = tmp_path / "test.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)

    mock_play = mocker.patch("sounddevice.play")
    mocker.patch("sounddevice.wait")

    player = Player()
    player.play(str(wav))
    mock_play.assert_called_once()


def test_player_raises_on_missing_file():
    player = Player()
    with pytest.raises(AudioOutputError, match="Failed to read"):
        player.play("/nonexistent/path/audio.wav")


def test_player_raises_on_portaudio_error(mocker, tmp_path):
    import sounddevice as sd
    wav = tmp_path / "test.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)

    mocker.patch("sounddevice.play", side_effect=sd.PortAudioError("no speaker"))
    mocker.patch("sounddevice.wait")

    player = Player()
    with pytest.raises(AudioOutputError, match="Playback failed"):
        player.play(str(wav))
