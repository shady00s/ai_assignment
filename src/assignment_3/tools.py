tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Supports basic arithmetic operations (+, -, *, /, **) and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g., '15 * 8 + 42' or '(100 - 32) * 5/9'"
                    }
                },
                "required": ["expression"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in a specific timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The IANA timezone name (e.g., 'UTC', 'US/Eastern', 'Europe/London'). Defaults to UTC."
                    }
                },
                "required": ["timezone"],
                "additionalProperties": False
            }
        }
    }
]