tools = [
    {
        "type": "function",
        "function": {
            "name": "run_local_command",
            "description": "Execute a local command on the system to perform tasks such as file manipulation, retrieving system information, or running scripts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The specific command to execute on the local system.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_research",
            "description": "Perform a web research query to gather information from online sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research query to perform.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze an image from a provided URL or a local path and generate a description of the image's content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "The URL or local path of the image to analyze.",
                    }
                },
                "required": ["image_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_os_default_calendar",
            "description": "Check the calendar for today or create a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check or create an event for (YYYY-MM-DD). Defaults to today if not provided.",
                    },
                    "time": {
                        "type": "string",
                        "description": "The time for the event (HH:MM). Optional.",
                    },
                    "event_title": {
                        "type": "string",
                        "description": "The title of the event. Optional.",
                    },
                    "event_description": {
                        "type": "string",
                        "description": "The description of the event. Optional.",
                    },
                },
                "required": [],
            },
        },
    },
]