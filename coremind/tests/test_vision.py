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

def _install_fake_cv2(monkeypatch, *, opened=True, frame=None):
    """Install a minimal fake `cv2` module into sys.modules and return it."""
    if frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self, index):
            self.index = index

        def isOpened(self):
            return opened

        def set(self, prop, value):
            return True

        def read(self):
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
