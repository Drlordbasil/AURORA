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
            "name": "do_nothing",
            "description": "A tool that does nothing. use this to skip tool use if no tool is needed. IE: hi or any normal conversation.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
