from __future__ import annotations

import asyncio
import base64
import binascii
import logging

from coremind.node_mcp.tools.camera_capture import ERROR_PREFIX
from coremind.tools.registry import Tool

logger = logging.getLogger(__name__)


def looks_like_jpeg_base64(s: str) -> bool:
    """Cheap sanity check that a tool result is a base64 image and not an error string."""
    if not s or s.startswith(ERROR_PREFIX) or len(s) < 64:
        return False
    try:
        head = base64.b64decode(s[:16], validate=True)
    except (binascii.Error, ValueError):
        return False
    return head[:2] == b"\xff\xd8"  # JPEG SOI marker


class LookAtSceneTool(Tool):
    """Capture a frame from the Node's camera and describe it with the Hub's vision model.

    The image is fetched from the Node over MCP (``capture_image``) and interpreted by a
    local Ollama vision model inside this tool — only the resulting text is returned to the
    orchestrating LLM, so the generic tool loop stays image-free.
    """

    name = "look"
    description = (
        "Capture an image from the camera and describe what is currently visible. "
        "Use this whenever the user asks what you can see, what's in the room, what "
        "something looks like, or to read or identify a physical object in view."
    )
    requires_confirmation = False
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Optional: what to look for or focus on (e.g. 'is the window open?'). "
                    "Defaults to a general description of the scene."
                ),
            },
        },
    }

    # MCP tool(s) this built-in wraps internally — hidden from the LLM by the dispatcher.
    wraps_mcp_tools = ("capture_image",)

    _DEFAULT_PROMPT = "Describe what you see in this image in one or two concise sentences."

    def __init__(self, dispatcher, vision_client) -> None:
        self._dispatcher = dispatcher
        self._vision = vision_client

    def run(self, **kwargs) -> str:
        raise NotImplementedError("The 'look' tool requires async execution (use run_async).")

    async def run_async(self, prompt: str = "", **kwargs) -> str:
        frame_b64 = await self._dispatcher.execute_async("capture_image", {})
        if not looks_like_jpeg_base64(frame_b64):
            logger.warning("look: no usable frame from Node — %s", frame_b64[:120])
            return (
                "I couldn't get an image from the camera. Make sure the Node is online "
                "and its camera is enabled."
            )
        question = prompt.strip() or self._DEFAULT_PROMPT
        # Vision inference can take several seconds — keep it off the event loop.
        return await asyncio.to_thread(self._vision.describe_image, frame_b64, question)
