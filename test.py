from openai import OpenAI
import json
import os
from Brain_modules.define_tools import tools
from Brain_modules.image_vision import ImageVision
from Brain_modules.tool_call_functions.web_research import WebResearchTool

client = OpenAI(base_url="http://localhost:11434/v1", api_key="sk-1234567890abcdef") # ollama api client
#client = OpenAI() # openai api client
# client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key = os.environ.get("GROQ_API_KEY"))


model = "llama3:instruct" # ollama model
#model = "llama3-70b-8192" # groq model
def send_web_research_tool(query):
    tool = WebResearchTool()
    response = tool.web_research(query)
    return response


def run_conversation(prompt):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": prompt}]
    tools = tools
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    print(response)
    response_message = response.choices[0].message
    print(response_message)
    tool_calls = response_message.tool_calls
    print(tool_calls)
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        print("tool_calls")

        available_functions = {
            
            "send_web_research_tool": send_web_research_tool,
        }  # only one function in this example, but you can have multiple
        print(available_functions)
        messages.append(response_message)  # extend conversation with assistant's reply
        print(messages)
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args),
            
            print(function_response)
            print(function_name)
            print(tool_call.id)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response.choices[0].message.content
print(run_conversation())