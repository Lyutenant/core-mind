from __future__ import annotations

import asyncio
import base64
import sys
import types

import numpy as np
import pytest

from coremind import VisionError
from coremind.vision.base import MockCamera


# ---------------------------------------------------------------------------
# MockCamera
# ---------------------------------------------------------------------------

def test_mock_camera_returns_jpeg_bytes():
    jpeg = MockCamera().capture_jpeg()
    assert isinstance(jpeg, bytes)
    assert jpeg[:2] == b"\xff\xd8"  # JPEG SOI marker


# ---------------------------------------------------------------------------
# OpenCVCamera (with a faked cv2 module — no real webcam needed)
# ---------------------------------------------------------------------------

def _bright_frame():
    # Mid-gray frame: mean brightness well above the dark threshold.
    return np.full((720, 1280, 3), 128, dtype=np.uint8)


def _dark_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _install_fake_cv2(monkeypatch, *, opened=True, frame=None, frames=None):
    """Install a minimal fake `cv2` module into sys.modules and return it.

    Pass `frames` (a list) to simulate a warm-up sequence; each read() returns
    the next frame and repeats the last one once exhausted.
    """
    if frame is None and frames is None:
        frame = _bright_frame()
    seq = list(frames) if frames is not None else None

    class _Cap:
        def __init__(self, index):
            self.index = index
            self._i = 0

        def isOpened(self):
            return opened

        def set(self, prop, value):
            return True

        def read(self):
            if seq is not None:
                f = seq[min(self._i, len(seq) - 1)]
                self._i += 1
                return (True, f)
            return (True, frame)

        def release(self):
            pass

    fake = types.ModuleType("cv2")
    fake.VideoCapture = _Cap
    fake.CAP_PROP_FRAME_WIDTH = 3
    fake.CAP_PROP_FRAME_HEIGHT = 4
    fake.IMWRITE_JPEG_QUALITY = 1
    fake.INTER_AREA = 3
    fake.resize = lambda f, size, interpolation=0: np.zeros((size[1], size[0], 3), dtype=np.uint8)
    fake.imencode = lambda ext, f, params: (True, np.frombuffer(b"\xff\xd8\xff\xe0jpegdata", dtype=np.uint8))
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return fake


def test_opencv_camera_captures_and_encodes(monkeypatch):
    _install_fake_cv2(monkeypatch)
    from coremind.vision.opencv_camera import OpenCVCamera

    cam = OpenCVCamera(camera_index=0, width=1280, height=720)
    jpeg = cam.capture_jpeg()
    assert jpeg[:2] == b"\xff\xd8"


def test_opencv_camera_skips_dark_warmup_frames(monkeypatch):
    # First frames are black (sensor warming up), then a real frame arrives.
    _install_fake_cv2(monkeypatch, frames=[_dark_frame(), _dark_frame(), _bright_frame()])
    from coremind.vision.opencv_camera import OpenCVCamera

    cam = OpenCVCamera(camera_index=0)
    captured = {}

    def _capture(f):
        captured["frame"] = f
        return f

    cam._downscale = _capture
    cam.capture_jpeg()
    assert float(captured["frame"].mean()) >= cam.dark_threshold  # skipped the black frames


def test_opencv_camera_returns_last_frame_when_all_dark(monkeypatch):
    # A genuinely dark room: never brightens — return the last frame after warmup.
    _install_fake_cv2(monkeypatch, frame=_dark_frame())
    from coremind.vision.opencv_camera import OpenCVCamera

    cam = OpenCVCamera(camera_index=0, warmup_seconds=0.05)
    jpeg = cam.capture_jpeg()
    assert jpeg[:2] == b"\xff\xd8"


def test_opencv_camera_raises_when_device_unavailable(monkeypatch):
    _install_fake_cv2(monkeypatch, opened=False)
    from coremind.vision.opencv_camera import OpenCVCamera

    cam = OpenCVCamera(camera_index=9)
    with pytest.raises(VisionError):
        cam.capture_jpeg()


def test_opencv_camera_missing_dependency(monkeypatch):
    # Simulate opencv not installed: importing cv2 raises ImportError.
    monkeypatch.setitem(sys.modules, "cv2", None)
    from coremind.vision.opencv_camera import OpenCVCamera

    with pytest.raises(VisionError, match="opencv"):
        OpenCVCamera()


def _capture_source(monkeypatch):
    """Install fake cv2, record the VideoCapture source argument."""
    fake = _install_fake_cv2(monkeypatch)
    captured = {}
    orig = fake.VideoCapture

    def _rec(src):
        captured["src"] = src
        return orig(src)

    fake.VideoCapture = _rec
    return captured


def test_opencv_camera_uses_device_path_over_index(monkeypatch):
    captured = _capture_source(monkeypatch)
    from coremind.vision.opencv_camera import OpenCVCamera

    path = "/dev/v4l/by-id/usb-Foo_Webcam-video-index0"
    OpenCVCamera(camera_index=3, camera_device=path).capture_jpeg()
    assert captured["src"] == path


def test_opencv_camera_uses_index_when_no_device(monkeypatch):
    captured = _capture_source(monkeypatch)
    from coremind.vision.opencv_camera import OpenCVCamera

    OpenCVCamera(camera_index=2).capture_jpeg()
    assert captured["src"] == 2


# ---------------------------------------------------------------------------
# Camera auto-selection (camera_select)
# ---------------------------------------------------------------------------

def test_list_camera_devices_reads_by_id(tmp_path, monkeypatch):
    from coremind.vision import camera_select

    (tmp_path / "usb-Foo-video-index0").write_text("")
    (tmp_path / "usb-Foo-video-index1").write_text("")
    monkeypatch.setattr(camera_select, "_BY_ID_DIR", tmp_path)
    devices = camera_select.list_camera_devices()
    assert devices == [str(tmp_path / "usb-Foo-video-index0"), str(tmp_path / "usb-Foo-video-index1")]


def test_list_camera_devices_empty_when_dir_missing(tmp_path, monkeypatch):
    from coremind.vision import camera_select

    monkeypatch.setattr(camera_select, "_BY_ID_DIR", tmp_path / "nope")
    assert camera_select.list_camera_devices() == []


def test_auto_select_camera_picks_index0(monkeypatch):
    from coremind import device_cache
    from coremind.vision import camera_select

    monkeypatch.setattr(camera_select, "list_camera_devices",
                        lambda: ["/dev/v4l/by-id/cam-video-index1", "/dev/v4l/by-id/cam-video-index0"])
    monkeypatch.setattr(device_cache, "get", lambda key: None)
    assert camera_select.auto_select_camera() == "/dev/v4l/by-id/cam-video-index0"


def test_resolve_camera_source_precedence(monkeypatch):
    from coremind import device_cache
    from coremind.vision import camera_select

    monkeypatch.setattr(device_cache, "remember", lambda *a: None)
    monkeypatch.setattr(camera_select, "auto_select_camera", lambda: "/dev/v4l/by-id/auto-index0")

    # Explicit path wins over everything.
    assert camera_select.resolve_camera_source("/dev/explicit", 5) == "/dev/explicit"
    # Explicit index wins over auto.
    assert camera_select.resolve_camera_source(None, 5) == 5
    # Falls back to auto-selection when both unset.
    assert camera_select.resolve_camera_source(None, None) == "/dev/v4l/by-id/auto-index0"


def test_resolve_camera_source_falls_back_to_zero(monkeypatch):
    from coremind import device_cache
    from coremind.vision import camera_select

    monkeypatch.setattr(device_cache, "remember", lambda *a: None)
    monkeypatch.setattr(camera_select, "auto_select_camera", lambda: None)
    assert camera_select.resolve_camera_source(None, None) == 0


# ---------------------------------------------------------------------------
# VisionConfig defaults
# ---------------------------------------------------------------------------

def test_vision_config_defaults():
    from coremind.config.settings import VisionConfig

    cfg = VisionConfig()
    assert cfg.camera_index is None      # None → auto-select
    assert cfg.camera_device is None


def test_vision_config_rejects_negative_index():
    from coremind.config.settings import VisionConfig

    with pytest.raises(ValueError):
        VisionConfig(camera_index=-1)


# ---------------------------------------------------------------------------
# Node capture_image tool
# ---------------------------------------------------------------------------

def test_capture_image_mock_provider_returns_base64():
    from coremind.node_mcp.tools import camera_capture

    camera_capture.init_camera(provider="mock")
    result = camera_capture.capture_image()
    assert not result.startswith(camera_capture.ERROR_PREFIX)
    decoded = base64.b64decode(result)
    assert decoded[:2] == b"\xff\xd8"


def test_capture_image_reports_error_string(monkeypatch):
    from coremind.node_mcp.tools import camera_capture

    def _boom():
        raise VisionError("no camera")

    monkeypatch.setattr(camera_capture, "_build_camera", lambda: types.SimpleNamespace(capture_jpeg=_boom))
    result = camera_capture.capture_image()
    assert result.startswith(camera_capture.ERROR_PREFIX)


# ---------------------------------------------------------------------------
# OllamaClient.describe_image
# ---------------------------------------------------------------------------

def test_describe_image_sends_images_field(monkeypatch):
    from coremind.brain.ollama_client import OllamaClient

    captured = {}

    def _fake_post(self, payload):
        captured["payload"] = payload
        return {"message": {"content": "a mug on a desk"}}

    monkeypatch.setattr(OllamaClient, "_post", _fake_post)
    client = OllamaClient(base_url="http://x:11434", model="llava:7b")
    out = client.describe_image("BASE64DATA", "what is this?")

    assert out == "a mug on a desk"
    msg = captured["payload"]["messages"][0]
    assert msg["images"] == ["BASE64DATA"]
    assert captured["payload"]["model"] == "llava:7b"


# ---------------------------------------------------------------------------
# LookAtSceneTool
# ---------------------------------------------------------------------------

class _FakeDispatcher:
    def __init__(self, result):
        self._result = result
        self.calls = []

    async def execute_async(self, name, args):
        self.calls.append((name, args))
        return self._result


class _FakeVision:
    def __init__(self):
        self.calls = []

    def describe_image(self, image_b64, prompt):
        self.calls.append((image_b64, prompt))
        return "a tidy desk"


def _valid_frame_b64() -> str:
    return base64.b64encode(MockCamera().capture_jpeg()).decode("ascii")


def test_look_tool_describes_frame():
    from coremind.tools.built_in.vision_tool import LookAtSceneTool

    disp = _FakeDispatcher(_valid_frame_b64())
    vision = _FakeVision()
    tool = LookAtSceneTool(dispatcher=disp, vision_client=vision)

    out = asyncio.run(tool.run_async(prompt="what's here?"))
    assert out == "a tidy desk"
    assert disp.calls == [("capture_image", {})]
    assert vision.calls[0][1] == "what's here?"


def test_look_tool_handles_camera_error():
    from coremind.tools.built_in.vision_tool import LookAtSceneTool

    disp = _FakeDispatcher("CAMERA_ERROR: no camera")
    tool = LookAtSceneTool(dispatcher=disp, vision_client=_FakeVision())

    out = asyncio.run(tool.run_async())
    assert "couldn't get an image" in out.lower()


def test_look_tool_run_sync_unsupported():
    from coremind.tools.built_in.vision_tool import LookAtSceneTool

    tool = LookAtSceneTool(dispatcher=_FakeDispatcher(""), vision_client=_FakeVision())
    with pytest.raises(NotImplementedError):
        tool.run()


# ---------------------------------------------------------------------------
# Dispatcher: look registration + hiding the wrapped MCP tool
# ---------------------------------------------------------------------------

def test_look_skipped_without_vision_client():
    from coremind.tools.dispatcher import ToolDispatcher

    d = ToolDispatcher()
    d.register_built_ins(["look"], vision_client=None)
    assert "look" not in d._tools


def test_capture_image_hidden_from_llm():
    from coremind.tools.dispatcher import ToolDispatcher

    d = ToolDispatcher()
    d.register_built_ins(["look"], vision_client=_FakeVision())

    fake_mcp = types.SimpleNamespace(
        _schemas=[
            {"type": "function", "function": {"name": "capture_image", "parameters": {}}},
            {"type": "function", "function": {"name": "set_volume", "parameters": {}}},
        ],
        tool_to_server={"capture_image": "node", "set_volume": "node"},
    )
    d.set_mcp_manager(fake_mcp)

    names = [s["function"]["name"] for s in d.get_tool_definitions()]
    assert "look" in names            # the built-in is advertised
    assert "set_volume" in names      # other MCP tools still advertised
    assert "capture_image" not in names  # the wrapped camera tool is hidden


def test_capture_image_hidden_even_without_look():
    # Node camera enabled but the Hub never registered `look` (no vision_model, or
    # `look` omitted from built_in): capture_image must still be hidden from the LLM.
    from coremind.tools.dispatcher import ToolDispatcher

    d = ToolDispatcher()  # no 'look' registered
    fake_mcp = types.SimpleNamespace(
        _schemas=[
            {"type": "function", "function": {"name": "capture_image", "parameters": {}}},
            {"type": "function", "function": {"name": "set_volume", "parameters": {}}},
        ],
        tool_to_server={"capture_image": "node", "set_volume": "node"},
    )
    d.set_mcp_manager(fake_mcp)

    names = [s["function"]["name"] for s in d.get_tool_definitions()]
    assert "capture_image" not in names
    assert "set_volume" in names


# ---------------------------------------------------------------------------
# Hub dashboard snapshot endpoints (/api/vision/capture, /api/vision/describe)
# Skipped where FastAPI isn't installed (server deps live on the Mac Mini Hub).
# ---------------------------------------------------------------------------

def _vision_test_client(monkeypatch, *, frame, has_camera=True, vision_model="llava:7b"):
    """Build a TestClient with the capture/vision plumbing stubbed out."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from coremind.server import app as app_module

    monkeypatch.setattr(app_module, "_get_dispatcher", lambda: _FakeDispatcher(frame))
    monkeypatch.setattr(
        app_module, "_mcp_manager",
        types.SimpleNamespace(tool_to_server={"capture_image": "node"} if has_camera else {}),
    )
    settings = types.SimpleNamespace(
        ollama=types.SimpleNamespace(base_url="http://x", vision_model=vision_model),
        brain=types.SimpleNamespace(timeout_seconds=30),
    )
    monkeypatch.setattr(app_module, "_get_settings", lambda: settings)
    return TestClient(app_module.app)


def test_capture_endpoint_returns_data_url(monkeypatch):
    client = _vision_test_client(monkeypatch, frame=_valid_frame_b64())
    resp = client.post("/api/vision/capture")
    assert resp.status_code == 200
    assert resp.json()["image"].startswith("data:image/jpeg;base64,")


def test_capture_endpoint_409_when_camera_unavailable(monkeypatch):
    client = _vision_test_client(monkeypatch, frame=_valid_frame_b64(), has_camera=False)
    resp = client.post("/api/vision/capture")
    assert resp.status_code == 409


def test_capture_endpoint_502_on_camera_error(monkeypatch):
    client = _vision_test_client(monkeypatch, frame="CAMERA_ERROR: no device")
    resp = client.post("/api/vision/capture")
    assert resp.status_code == 502
    assert "no device" in resp.json()["error"]


def test_describe_endpoint_409_without_vision_model(monkeypatch):
    client = _vision_test_client(monkeypatch, frame=_valid_frame_b64(), vision_model=None)
    resp = client.post("/api/vision/describe", json={"image": _valid_frame_b64()})
    assert resp.status_code == 409


def test_describe_endpoint_describes_frame(monkeypatch):
    client = _vision_test_client(monkeypatch, frame=_valid_frame_b64())
    from coremind.brain import ollama_client

    class _StubClient:
        def __init__(self, **kwargs):
            pass

        def describe_image(self, image_b64, prompt):
            return "a tidy desk"

    monkeypatch.setattr(ollama_client, "OllamaClient", _StubClient)
    resp = client.post(
        "/api/vision/describe",
        json={"image": "data:image/jpeg;base64," + _valid_frame_b64()},
    )
    assert resp.status_code == 200
    assert resp.json()["description"] == "a tidy desk"
