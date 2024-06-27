import json
import os
import subprocess
import time
import requests
from datetime import datetime
from openai import OpenAI
from groq import Groq
import ollama
import tiktoken
from Brain_modules.image_vision import ImageVision
from Brain_modules.tool_call_functions.web_research import WebResearchTool
from Brain_modules.define_tools import tools
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential

MAX_TOKENS_PER_MINUTE = 5500  # Reduced from 6000 to provide a buffer
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
from Brain_modules.final_agent_persona import FinalAgentPersona

def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class LLM_API_Calls:
    def __init__(self):
        self.client = None
        self.model = None
        self.setup_client()
        self.image_vision = ImageVision()
        self.chat_history = []
        self.max_tokens = 4000
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.web_research_tool = WebResearchTool()
        self.tokens_used = 0
        self.start_time = time.time()
        self.available_functions = {
            "run_local_command": self.run_local_command,
            "web_research": self.web_research_tool.web_research,
            "do_nothing": self.do_nothing
        }
        self.interaction_count = 0
        self.max_interactions = 10
        self.rate_limit_remaining = MAX_TOKENS_PER_MINUTE
        self.rate_limit_reset = time.time() + 60

    def setup_client(self):
        try:
            self.client, self.model = self.choose_API_provider()
        except Exception as e:
            print(f"Error setting up client: {e}")
            raise

    def choose_API_provider(self):
        llm = os.environ.get("LLM_PROVIDER", "Groq")
        if llm == "OpenAI":
            api_key = os.environ.get("OPENAI_API_KEY") or input("Enter your OpenAI API key: ").strip()
            model = os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")
            client = OpenAI(api_key=api_key)
        elif llm == "ollama":
            model = os.environ.get("OLLAMA_MODEL", "llama3:instruct")
            client = ollama
        else:  # Default to Groq
            api_key = os.environ.get("GROQ_API_KEY") or input("Enter your Groq API key: ").strip()
            model = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
            client = Groq(api_key=api_key)
        return client, model

    def count_tokens(self, text):
        return len(self.encoding.encode(str(text)))

    def truncate_text(self, text, max_tokens):
        tokens = self.encoding.encode(str(text))
        return self.encoding.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else text

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, max=10))
    def run_local_command(self, command, progress_callback=None):
        if progress_callback:
            progress_callback(f"Executing local command: {command}")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            output = result.stdout
            if progress_callback:
                progress_callback(f"Local command executed successfully")
            return {"command": command, "output": output, "datetime": get_current_datetime()}
        except subprocess.CalledProcessError as e:
            if progress_callback:
                progress_callback(f"Error executing local command: {e.stderr}")
            return {"command": command, "error": f"Command execution failed: {e.stderr}", "datetime": get_current_datetime()}
        except Exception as e:
            if progress_callback:
                progress_callback(f"Unexpected error during local command execution: {str(e)}")
            return {"command": command, "error": str(e), "datetime": get_current_datetime()}

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, max=10))
    def analyze_image(self, image_url, progress_callback=None):
        if progress_callback:
            progress_callback(f"Analyzing image: {image_url}")
        try:
            if os.path.exists(image_url):
                description = self.image_vision.analyze_image(image_url)
            elif image_url.startswith("http"):
                if progress_callback:
                    progress_callback("Downloading image from URL")
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                temp_file = 'temp_image.jpg'
                with open(temp_file, 'wb') as file:
                    file.write(response.content)
                description = self.image_vision.analyze_image(temp_file)
                os.remove(temp_file)
            else:
                raise ValueError("Invalid image URL or path")
            if progress_callback:
                progress_callback("Image analysis completed")
            return {"image_url": image_url, "description": description, "datetime": get_current_datetime()}
        except RequestException as e:
            if progress_callback:
                progress_callback(f"Failed to fetch image: {str(e)}")
            return {"error": f"Failed to fetch image: {str(e)}", "datetime": get_current_datetime()}
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error during image analysis: {str(e)}")
            return {"error": str(e), "datetime": get_current_datetime()}

    def do_nothing(self, progress_callback=None):
        if progress_callback:
            progress_callback("Executing do nothing tool")
        return {"result": "Nothing was done", "datetime": get_current_datetime()}

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, max=10))
    def chat(self, prompt, progress_callback=None):
        self.reset_token_usage()

        if progress_callback:
            progress_callback("Preparing chat messages")

        messages = [
            {"role": "system", "content": f"""
             #####CHAT HISTORY START####You are {FinalAgentPersona.name}. {FinalAgentPersona.description} 
             # who has access to function calling and here is which ones you can use and how to get creative 
             # {self.chat_history[-1]['content'] if self.chat_history else ''}
             #####CHAT HISTORY END####
             """},
            {"role": "assistant", "content": self.chat_history[-1]['content'] if self.chat_history else "Hello! How can I help you today? I have various tool capabilites to assist you."},
            {"role": "user", "content": prompt},
        ]

        try:
            self.check_rate_limit()
            
            if progress_callback:
                progress_callback("Sending request to language model")
            
            if isinstance(self.client, OpenAI) or isinstance(self.client, Groq):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=self.max_tokens
                )
                self.update_rate_limit(getattr(response, 'headers', {}))
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

            elif self.client == ollama:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                response_message = response['message']
                tool_calls = response_message.get('tool_calls', [])

            if progress_callback:
                progress_callback("Received response from language model")

            self.chat_history.append({"role": "assistant", "content": response_message.content})
            self.update_token_usage(messages, response_message.content)

            if tool_calls:
                if progress_callback:
                    progress_callback("Processing tool calls")
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = self.available_functions.get(function_name)
                    if function_to_call is None:
                        raise ValueError(f"Unknown function: {function_name}")
                    function_args = json.loads(tool_call.function.arguments)
                    if progress_callback:
                        progress_callback(f"Executing function: {function_name}")
                    function_response = function_to_call(**function_args, progress_callback=progress_callback)
                    
                    tool_response = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response)
                    }
                    self.chat_history.append(tool_response)
                    self.update_token_usage(messages, json.dumps(function_response))

                messages.extend(self.chat_history[-2:])  # Add assistant's response and tool response

                if progress_callback:
                    progress_callback("Generating final response based on tool results")
                second_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens
                ) if isinstance(self.client, (OpenAI, Groq)) else self.client.chat(
                    model=self.model,
                    messages=messages
                )

                second_response_content = second_response.choices[0].message.content if isinstance(self.client, (OpenAI, Groq)) else second_response['message']['content']
                self.chat_history.append({"role": "assistant", "content": second_response_content})
                self.update_token_usage(messages, second_response_content)
                
                if progress_callback:
                    progress_callback("Final response generated")
                return second_response_content
            else:
                if progress_callback:
                    progress_callback("Response generated without tool use")
                return response_message.content

        except Exception as e:
            error_message = f"Error in chat: {str(e)}"
            if progress_callback:
                progress_callback(f"Error occurred: {error_message}")
            print(error_message)
            return {"error": error_message, "datetime": get_current_datetime()}

    def update_token_usage(self, messages, response):
        tokens_used = sum(self.count_tokens(msg["content"]) for msg in messages) + self.count_tokens(response)
        self.tokens_used += tokens_used
        self.rate_limit_remaining -= tokens_used

    def reset_token_usage(self):
        current_time = time.time()
        if current_time >= self.rate_limit_reset:
            self.tokens_used = 0
            self.rate_limit_remaining = MAX_TOKENS_PER_MINUTE
            self.rate_limit_reset = current_time + 60

    def check_rate_limit(self):
        if self.rate_limit_remaining <= 0:
            sleep_time = max(0, self.rate_limit_reset - time.time())
            time.sleep(sleep_time)
            self.reset_token_usage()

    def update_rate_limit(self, headers):
        remaining = headers.get('X-RateLimit-Remaining')
        reset = headers.get('X-RateLimit-Reset')
        
        if remaining is not None:
            try:
                self.rate_limit_remaining = int(remaining)
            except ValueError:
                self.rate_limit_remaining = MAX_TOKENS_PER_MINUTE
        
        if reset is not None:
            try:
                self.rate_limit_reset = float(reset)
            except ValueError:
                self.rate_limit_reset = time.time() + 60
