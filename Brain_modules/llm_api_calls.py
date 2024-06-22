import json
import os
import subprocess
import time
from openai import OpenAI
from groq import Groq
import ollama
from datetime import datetime
from kivy.clock import Clock
import tiktoken
from Brain_modules.image_vision import ImageVision
from Brain_modules.tool_call_functions.web_research import WebResearchTool
from Brain_modules.define_tools import tools

# Constants for token limits
MAX_TOKENS_PER_MINUTE = 6000
API_CALL_MAX_TOKENS = 800

def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class LLM_API_Calls:
    def __init__(self, status_update_callback):
        self.client = None
        self.model = None
        self.setup_client()
        self.image_vision = ImageVision()
        self.status_update_callback = status_update_callback
        self.chat_history = []
        self.max_tokens = 4000
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.web_research_tool = WebResearchTool(status_update_callback, self.chat, max_tokens=self.max_tokens)
        self.tokens_used = 0
        self.start_time = time.time()
        self.available_functions = {
            "run_local_command": self.run_local_command,
            "web_research": self.web_research_tool.web_research,
            "analyze_image": self.analyze_image,
            "check_os_default_calendar": self.check_os_default_calendar
        }

    def _update_status(self, message):
        Clock.schedule_once(lambda dt: self.status_update_callback(message), 0)

    def setup_client(self):
        try:
            self.client, self.model = self.choose_API_provider()
            self._update_status("Client setup successful.")
        except Exception as e:
            print(f"Error setting up client: {e}")
            self._update_status(f"Error setting up client: {e}")

    def choose_API_provider(self):
        llm = "Groq"
        if llm == "OpenAI":
            api_key = os.environ.get("OPENAI_API_KEY") or input("Enter your OpenAI API key: ").strip()
            model = "gpt-4"
            client = OpenAI(api_key=api_key)
            return client, model
        elif llm == "ollama":
            model = "llama3:instruct"
            return ollama, model
        elif llm == "Groq":
            api_key = os.environ.get("GROQ_API_KEY") or input("Enter your Groq API key: ").strip()
            model = "llama3-70b-8192"
            client = Groq(api_key=api_key)
            return client, model
        else:
            raise ValueError("Invalid API provider selected")

    def count_tokens(self, text):
        return len(self.encoding.encode(text))

    def truncate_text(self, text, max_tokens):
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])

    def chunk_text(self, text, max_tokens):
        tokens = self.encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk = self.encoding.decode(tokens[i:i+max_tokens])
            chunks.append(chunk)
        return chunks

    def check_os_default_calendar(self, date=None, time=None, event_title=None, event_description=None):
        self._update_status(f"Checking calendar for {date or 'today'}")
        try:
            if date and event_title:
                command = f'powershell.exe New-Event -Title "{event_title}"'
                if time:
                    command += f' -StartDate "{date}T{time}:00"'
                else:
                    command += f' -StartDate "{date}"'
                if event_description:
                    command += f' -Description "{event_description}"'
                subprocess.run(command, shell=True)
                output = f"Event '{event_title}' created on {date}."
            else:
                command = 'powershell.exe Get-Event | Where-Object { $_.StartDate -eq [datetime]::Today }'
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                output = result.stdout
            self._update_status(f"Calendar check complete: {output}")
            return json.dumps({"output": output, "datetime": get_current_datetime()})
        except Exception as e:
            error_message = json.dumps({"message": f"Error executing command: {str(e)}", "datetime": get_current_datetime()})
            self._update_status(f"Error checking calendar: {error_message}")
            return json.dumps({"error": error_message, "datetime": get_current_datetime()})

    def run_local_command(self, command):
        self._update_status(f"Executing command: {command}")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout
            error = result.stderr
            if error:
                output = json.dumps({"message": f"Command executed with errors: {error}", "datetime": get_current_datetime()})
            else:
                output = json.dumps({"message": f"Command executed successfully: {output}", "datetime": get_current_datetime()})
            self._update_status(f"Local command executed: {output}")
            return json.dumps({"command": command, "output": output, "error": error, "datetime": get_current_datetime()})
        except Exception as e:
            error_message = json.dumps({"message": f"Error executing command: {str(e)}", "datetime": get_current_datetime()})
            self._update_status(f"Error executing local command: {error_message}")
            return json.dumps({"command": command, "error": error_message, "datetime": get_current_datetime()})

    def analyze_image(self, image_url):
        try:
            if os.path.exists(image_url):
                description = self.image_vision.analyze_image_local(image_url)
            else:
                description = self.image_vision.analyze_image(image_url)
            self._update_status(f"Image analysis completed for {image_url}")
            return json.dumps({"image_url": image_url, "description": description, "datetime": get_current_datetime()})
        except Exception as e:
            error_message = json.dumps({"message": f"Error analyzing image: {str(e)}", "datetime": get_current_datetime()})
            self._update_status(f"Error analyzing image: {error_message}")
            return json.dumps({"image_url": image_url, "error": error_message, "datetime": get_current_datetime()})

    def chat(self, system_prompt, prompt):
        self.reset_token_usage()
        self.chat_history.append({"role": "user", "content": prompt})
        self._update_status("Starting chat process.")

        system_prompt = f"""
        {system_prompt}
        current time: {time.strftime('%H:%M:%S')}
        current date and time: {get_current_datetime()}
        you do things the most efficient way possible combining your tool usage and your knowledge of the world and how computers work.
        """

        messages = [{"role": "system", "content": system_prompt}] + self.chat_history

        # Count tokens and truncate if necessary
        total_tokens = sum(self.count_tokens(msg["content"]) for msg in messages)
        while total_tokens > self.max_tokens:
            if len(messages) > 1:
                removed_message = messages.pop(1)  # Remove the oldest non-system message
                total_tokens -= self.count_tokens(removed_message["content"])
            else:
                # If only system message remains, truncate it
                messages[0]["content"] = self.truncate_text(messages[0]["content"], self.max_tokens)
                total_tokens = self.count_tokens(messages[0]["content"])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=API_CALL_MAX_TOKENS
            )
            response_message = response.choices[0].message
            self._update_status("Chat response received.")
            tool_calls = response_message.tool_calls
            self.chat_history.append({"role": "assistant", "content": response_message.content})
            self.update_token_usage(messages, response_message.content)

            if tool_calls:
                available_functions = self.available_functions
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(**function_args)
                    self.chat_history.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response
                        }
                    )
                    self._update_status(f"Function '{function_name}' executed.")
                    self.update_token_usage(messages, function_response)

                # Prepare messages for the second API call
                second_messages = messages + [
                    {"role": "assistant", "content": response_message.content},
                    {"role": "tool", "content": json.dumps([{"name": tc.function.name, "response": self.chat_history[-1]["content"]} for tc in tool_calls])}
                ]

                # Truncate messages if necessary
                total_tokens = sum(self.count_tokens(msg["content"]) for msg in second_messages)
                while total_tokens > self.max_tokens:
                    if len(second_messages) > 1:
                        removed_message = second_messages.pop(1)
                        total_tokens -= self.count_tokens(removed_message["content"])
                    else:
                        second_messages[0]["content"] = self.truncate_text(second_messages[0]["content"], self.max_tokens)
                        total_tokens = self.count_tokens(second_messages[0]["content"])

                second_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=second_messages
                )
                second_response_content = second_response.choices[0].message.content
                self._update_status("Tool responses processed.")
                self.chat_history.append({"role": "assistant", "content": second_response_content})
                self.update_token_usage(second_messages, second_response_content)
                return second_response_content
            else:
                return response_message.content
        except Exception as e:
            error_message = f"Error in chat: {str(e)}"
            print(error_message)
            self._update_status(error_message)
            return json.dumps({"error": error_message, "datetime": get_current_datetime()})

    def update_token_usage(self, messages, response):
        tokens_used = sum(self.count_tokens(msg["content"]) for msg in messages) + self.count_tokens(response)
        self.tokens_used += tokens_used

    def reset_token_usage(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 60:
            self.tokens_used = 0
            self.start_time = current_time

    def ensure_token_limit(self):
        while self.tokens_used > MAX_TOKENS_PER_MINUTE:
            self.reset_token_usage()
            time.sleep(1)
