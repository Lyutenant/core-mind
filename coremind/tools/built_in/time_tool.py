from __future__ import annotations

import datetime
import logging
import zoneinfo

from coremind.tools.registry import Tool

logger = logging.getLogger(__name__)


class TimeTool(Tool):
    name = "get_current_time"
    description = "Get the current date and time."
    requires_confirmation = False
    parameters = {"type": "object", "properties": {}}

    def __init__(self, timezone: str | None = None) -> None:
        self._tz: zoneinfo.ZoneInfo | None = None
        if timezone:
            try:
                self._tz = zoneinfo.ZoneInfo(timezone)
            except (zoneinfo.ZoneInfoNotFoundError, KeyError):
                logger.warning(
                    "Unknown timezone %r — falling back to server local time. "
                    "Use an IANA name such as 'America/Los_Angeles'.",
                    timezone,
                )

    def run(self, **kwargs) -> str:
        now = datetime.datetime.now(tz=self._tz)
        return now.strftime("Today is %A, %B %d, %Y. The time is %I:%M %p.")
