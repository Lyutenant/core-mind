from __future__ import annotations

import datetime

from coremind.tools.registry import Tool


class TimeTool(Tool):
    name = "get_current_time"
    description = "Get the current date and time."
    requires_confirmation = False
    parameters = {"type": "object", "properties": {}}

    def run(self, **kwargs) -> str:
        now = datetime.datetime.now()
        return now.strftime("Today is %A, %B %d, %Y. The time is %I:%M %p.")
