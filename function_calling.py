# function_calling.py

import json
import subprocess
import os
import time
from groq import Groq
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

class FunctionCalling:
    """
    The FunctionCalling class integrates function calling capabilities into the AURORA chat chain.
    It allows the chat model to use external functions to enhance responses based on specific prompts.
    """

    MODEL = 'llama3-70b-8192'

    def __init__(self, api_key):
        """
        Initialize the FunctionCalling class with the provided API key.

        Args:
            api_key (str): The API key for accessing the Groq service.
        """
        self.client = Groq(api_key=api_key)

    def run_local_command(self, command):
        """
        Execute a local command on a Windows 11 PC and return the output.

        Args:
            command (str): The command to execute.

        Returns:
            str: The output of the command.
        """
        if command.lower() == "date":
            command = "date /T"
        elif command.lower() == "time":
            command = "time /T"
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print("<run_local_command> used by agent - Success")
            print(result.stdout)
            return json.dumps({"command": command, "output": result.stdout, "error": result.stderr})
        except Exception as e:
            print("<run_local_command> used by agent - Failed")
            print(str(e))
            return json.dumps({"command": command, "error": str(e)})

    def web_research(self, query):
        """
        Perform a web research query using Selenium and BeautifulSoup to extract useful information from Google.

        Args:
            query (str): The research query.

        Returns:
            str: The extracted useful information from Google search results.
        """
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        try:
            driver.get("https://www.google.com")
            search_box = driver.find_element(By.NAME, "q")
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)

            time.sleep(3)  # Increased wait time for search results to load

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            search_results = []

            for result in soup.select('.tF2Cxc'):
                title_element = result.select_one('.DKV0Md')
                snippet_element = result.select_one('.IsZvec')
                
                # Check if elements exist
                if title_element and snippet_element:
                    title = title_element.get_text()
                    snippet = snippet_element.get_text()
                    search_results.append({"title": title, "snippet": snippet})

            print("<web_research> used by agent - Success")
            for res in search_results:
                print(f"Title: {res['title']}, Snippet: {res['snippet']}")
            return json.dumps({"query": query, "results": search_results})
        except Exception as e:
            print("<web_research> used by agent - Failed")
            print(str(e))
            return json.dumps({"query": query, "error": str(e)})
        finally:
            driver.quit()

    def run_conversation(self, user_prompt):
        """
        Run a conversation with function calling capabilities.

        Args:
            user_prompt (str): The user's prompt to the chat model.

        Returns:
            str: The response from the chat model.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a function calling LLM that uses the data extracted from local commands and web research functions to provide detailed responses to the user."
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_local_command",
                    "description": "Execute a local command on the system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute (e.g. 'dir' or 'echo Hello World')",
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
                    "description": "Perform a web research query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The research query (e.g. 'Latest AI trends')",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls:
            available_functions = {
                "run_local_command": self.run_local_command,
                "web_research": self.web_research,
            }
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
                time.sleep(10)  # Sleep for 10 seconds between function calls
            second_response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages
            )
            time.sleep(10)  # Sleep for 10 seconds between responses
            return second_response.choices[0].message.content
        return response_message.content

if __name__ == "__main__":
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
    else:
        fc = FunctionCalling(api_key)
        
        # Test prompts for run_local_command function
        user_prompts = [
            "Please run a command to list directory contents.",
            "Execute the command to display the current date.",
            "Execute the command to display the current time.",
            "Show the current Python version."
        ]
        for prompt in user_prompts:
            print(f"Testing run_conversation with prompt: {prompt}")
            print(fc.run_conversation(prompt))
            time.sleep(10)  # Sleep for 10 seconds between tests
        
        # Test prompts for web_research function
        research_prompts = [
            "Search for the latest AI trends.",
            "Find information about Python 3.12 new features.",
            "Look up the benefits of using Docker."
        ]
        for prompt in research_prompts:
            print(f"Testing run_conversation with prompt: {prompt}")
            print(fc.run_conversation(prompt))
            time.sleep(10)  # Sleep for 10 seconds between tests
