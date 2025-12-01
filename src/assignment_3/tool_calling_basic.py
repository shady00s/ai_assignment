import os
import json

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

# Import tools from tools.py
from tools import tools

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
    
)

def calculate(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate a mathematical expression.
    """
    try:
        print(f"Evaluating expression: {expression}")
        # Security check: Allow only specific characters to prevent code injection
        allowed_chars = set("0123456789+-*/().% ")
        for char in expression:
            if char not in allowed_chars:
                return {"error": "Invalid character in expression", "success": False}
       

        result = eval(expression, {"__builtins__": None}, {})
        return {"result": result, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}
def convert_temeprature(value: float, from_unit: str, to_unit: str)-> Dict[str, Any]:
    """
    Convert temperature between celsius and fahrenheit and vice versa.
        Args:
        value: Temperature value
        from_unit: "celsius" or "fahrenheit"
        to_unit: "celsius" or "fahrenheit"
        Returns:
        dict with converted value and success flag
    """
    try:
        if from_unit == "celsius":
            if to_unit == "fahrenheit":
                result = (value * 9/5) + 32
            elif to_unit == "celsius":
                result = value
            else:
                return {"error": "Invalid to_unit", "success": False}
        elif from_unit == "fahrenheit":
            if to_unit == "celsius":
                result = (value - 32) * 5/9
            elif to_unit == "fahrenheit":
                result = value
            else:
                return {"error": "Invalid to_unit", "success": False}
        else:
            return {"error": "Invalid from_unit", "success": False}

        return {"result": result, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def get_current_time(timezone: str = "UTC") -> Dict[str, Any]:
    """
    Get current time in specified timezone.
    """
    try:
        # Get time with timezone info
        dt = datetime.now(ZoneInfo(timezone))
        return {
            "time": dt.strftime("%Y-%m-%d %H:%M:%S"), 
            "timezone": timezone, 
            "success": True
        }
    except Exception as e:
        return {"error": f"Invalid timezone or error: {str(e)}", "success": False}

# Define tool schemas


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute the requested tool with given arguments."""
    if tool_name == "calculate":
        return calculate(arguments.get("expression", ""))
    elif tool_name == "get_current_time":
        return get_current_time(arguments.get("timezone", "UTC"))
    elif tool_name == "convert_temperature":
        return convert_temeprature(arguments.get("value", ""), arguments.get("from_unit", ""), arguments.get("to_unit", ""))
    else:
        return {"error": f"Unknown tool: {tool_name}", "success": False}

def run_conversation(user_message: str) -> str:
    """
    Run complete tool calling workflow.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant, answer the following questions as best you can using the provided tools only. if you don't know the answer to a question, say 'I don't know'. when user asks for converting temprature or calculations, you must use calculate and convert_temperature tools."},
        {"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"  --> Calling tool: {function_name} with {function_args}")
            
            function_response = execute_tool(function_name, function_args)
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response)
            })

        second_response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=messages
        )
        return second_response.choices[0].message.content
    
    return response_message.content

def main():
    # Test queries
    test_queries = [
        "Convert temperature from 37 degrees Celsius to Fahrenheit",
        "What is 234 * 567?",
        "What time is it in UTC?",
        "calculate thirteen by thirteen",
        "What's the current time in  Cairo and hongkong?",  
        "What's the weather in Tokyo?",
        "Compare the weather in Paris and London",
        "What's the temperature in New York in Fahrenheit?",
        "Get weather for Berlin and convert the temperature to Fahrenheit",
        "Is it warmer in Miami or Seattle? By how much in Fahrenheit?",
    ]

    print("=" * 60)
    print("Multi-Tool Weather Assistant")    
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        try:
            answer = run_conversation(query)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
    print()

if __name__ == "__main__":
    main()