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
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from typing import Tuple, List, Any, Union, Dict
from Brain_modules.tool_call_functions.do_nothing import do_nothing
from Brain_modules.tool_call_functions.call_expert import call_expert

MAX_TOKENS_PER_MINUTE = 5500
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
from Brain_modules.final_agent_persona import FinalAgentPersona

def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class LLM_API_Calls:
    def __init__(self):
        self.client = None
        self.model = None
        self.current_api_provider = "ollama"
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
            "do_nothing": do_nothing,
            "analyze_image": self.analyze_image,
            "call_expert": call_expert
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
        if self.current_api_provider == "OpenAI":
            api_key = os.environ.get("OPENAI_API_KEY") or input("Enter your OpenAI API key: ").strip()
            model = os.environ.get("OPENAI_MODEL", "gpt-4o")
            client = OpenAI(api_key=api_key)
        elif self.current_api_provider == "ollama":
            api_key = "ollama"
            model = os.environ.get("OLLAMA_MODEL", "llama3:instruct")
            client = OpenAI(base_url="http://localhost:11434/v1", api_key=api_key)
        elif self.current_api_provider == "Groq":
            api_key = os.environ.get("GROQ_API_KEY") or input("Enter your Groq API key: ").strip()
            model = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        else:
            raise ValueError("Unsupported LLM Provider")
        return client, model

    def update_api_provider(self, provider):
        self.current_api_provider = provider
        self.setup_client()
        
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


    def chat(self, prompt: str, system_message: str, tools: List[dict], progress_callback=None) -> Tuple[Union[str, Dict[str, str]], List[Any]]:
        try:
            response, tool_calls = self._chat_with_retry(prompt, system_message, tools, progress_callback)
            if not response:
                response = "I apologize, but I couldn't generate a response. How else can I assist you?"
            return response, tool_calls
        except RetryError as e:
            error_message = f"Failed to get a response after {MAX_RETRIES} attempts: {str(e)}"
            if progress_callback:
                progress_callback(error_message)
            return error_message, []
        except Exception as e:
            error_message = f"Unexpected error in chat: {str(e)}"
            if progress_callback:
                progress_callback(error_message)
            return error_message, []

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, max=10))
    def _chat_with_retry(self, prompt: str, system_message: str, tools: List[dict], progress_callback=None) -> Tuple[Union[str, Dict[str, str]], List[Any]]:
        self.reset_token_usage()

        if progress_callback:
            progress_callback("Preparing chat messages")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": self.chat_history[-1]['content'] if self.chat_history else "Hello! How can I help you today? I have various tool capabilities to assist you."},
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
                tool_calls = response_message.tool_calls or []

            elif self.current_api_provider == "ollama":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls or []

            if progress_callback:
                progress_callback("Received response from language model")

            content = response_message.content
            self.chat_history.append({"role": "assistant", "content": content})
            self.update_token_usage(messages, content)

            return content, tool_calls

        except Exception as e:
            error_message = f"Error in chat: {str(e)}"
            if progress_callback:
                progress_callback(f"Error occurred: {error_message}")
            print(error_message)
            raise

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

llm_api_calls = LLM_API_Calls()