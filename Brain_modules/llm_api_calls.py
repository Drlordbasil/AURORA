import json
import os
import subprocess
import time
from traceback import extract_tb
from bs4 import BeautifulSoup
from openai import OpenAI, APIConnectionError, APIStatusError
from groq import Groq
import ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
import requests
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from Brain_modules.image_vision import ImageVision
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import chromadb

def test_python_code_on_windows_subprocess(script_raw):
    try:
        with open("temp_script.py", "w") as file:
            file.write(script_raw)
        result = subprocess.run(["python", "temp_script.py"], shell=True, capture_output=True, text=True)
        output = result.stdout
        error = result.stderr
        if error:
            output = json.dumps({"message": f"Script executed with errors: {error}"})
        else:
            output = json.dumps({"message": f"Script executed successfully: {output}"})
        return json.dumps({"script": script_raw, "output": output, "error": error})
    except Exception as e:
        error_message = json.dumps({"message": f"Error executing script: {str(e)}"})
        return json.dumps({"script": script_raw, "error": error_message})

chromadb_client = chromadb.Client()
collection = chromadb_client.create_collection(name="aurora_test_phase")

def time_now():
    return time.strftime("%H:%M:%S")
time_now = time_now()
print(time_now)

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
            "description": "Analyze an image from a provided URL and generate a description of the image's content.",
            "parameters": {
                        "type": "object",
                        "properties": {
                            "image_url": {
                                "type": "string",
                                "description": "The URL of the image to analyze.",
                            }
                        },
                        "required": ["image_url"],
                    },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_text_from_pdf",
            "description": "Extract text content from a PDF file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_url": {
                        "type": "string",
                        "description": "The URL of the PDF file.",
                    }
                },
                "required": ["pdf_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": "Analyze the sentiment of a given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze.",
                    }
                },
                "required": ["text"],
            },
        },
    }
]

def choose_API_provider():
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

class LLM_API_Calls:
    def __init__(self):
        self.client = None
        self.model = None
        self.setup_client()
        self.image_vision = ImageVision()
        self.available_functions = {
            "run_local_command": self.run_local_command,
            "web_research": self.web_research,
            "analyze_image": self.image_vision.analyze_image,
            "extract_text_from_pdf": self.extract_text_from_pdf,
            "analyze_sentiment": self.analyze_sentiment,
        }
        self.chat_history = []

    def run_local_command(self, command):
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout
            error = result.stderr
            if error:
                output = json.dumps({"message": f"Command executed with errors: {error}"})
            else:
                output = json.dumps({"message": f"Command executed successfully: {output}"})
            return json.dumps({"command": command, "output": output, "error": error})
        except Exception as e:
            error_message = json.dumps({"message": f"Error executing command: {str(e)}"})
            return json.dumps({"command": command, "error": error_message})

    def _initialize_webdriver(self):
        """Initialize and return a Chrome WebDriver with predefined options."""
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-gpu')
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver

    def extract_text_from_url(self, url):
        """Extract and return the main text content from a given URL."""
        driver = self._initialize_webdriver()
        try:
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            return text
        except Exception as e:
            return f"Error extracting text from {url}: {str(e)}"
        finally:
            driver.quit()

    def web_research(self, query):
        """Perform web research based on the given query and return aggregated content."""
        search_engines = ["https://www.google.com", "https://www.bing.com"]
        search_results = []

        for engine in search_engines:
            driver = self._initialize_webdriver()
            try:
                driver.get(engine)
                search_box = driver.find_element(By.NAME, "q")
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)

                soup = BeautifulSoup(driver.page_source, 'html.parser')
                results = soup.select('.tF2Cxc')[:2] if 'google' in engine else soup.select('.b_algo')[:2]

                for result in results:
                    title_element = result.select_one('.DKV0Md') if 'google' in engine else result.select_one('h2')
                    link_element = result.select_one('a')

                    if title_element and link_element:
                        title = title_element.get_text()
                        link = link_element.get('href')

                        main_content = self.extract_text_from_url(link)
                        search_results.append({
                            "title": title,
                            "link": link,
                            "content": main_content
                        })

                        # Fetch additional sub-links for more information
                        driver.get(link)
                        sub_soup = BeautifulSoup(driver.page_source, 'html.parser')
                        sub_links = sub_soup.select('a')[:1]
                        for sub_link in sub_links:
                            sub_url = sub_link.get('href')
                            if sub_url and sub_url.startswith('http'):
                                sub_content = self.extract_text_from_url(sub_url)
                                search_results.append({
                                    "title": f"Sublink from {title}",
                                    "link": sub_url,
                                    "content": sub_content
                                })
            except Exception as e:
                print(f"Error using {engine}: {e}")
            finally:
                driver.quit()

        # Aggregate and filter the content
        aggregated_content = []
        for result in search_results:
            sentences = sent_tokenize(result['content'])
            filtered_sentences = [sentence for sentence in sentences if TextBlob(sentence).sentiment.subjectivity < 0.5]
            aggregated_content.extend(filtered_sentences)

        final_content = ' '.join(aggregated_content)
        if len(final_content.split()) > 1500:
            final_content = ' '.join(final_content.split()[:1500])

        return json.dumps({"query": query, "results": final_content}) if final_content else json.dumps({"query": query, "error": "No results found."})

    def extract_text_from_pdf(self, pdf_url):
        try:
            response = requests.get(pdf_url)
            with open("temp.pdf", "wb") as file:
                file.write(response.content)
            text = extract_tb("temp.pdf")
            os.remove("temp.pdf")
            return json.dumps({"pdf_url": pdf_url, "text": text})
        except Exception as e:
            return json.dumps({"pdf_url": pdf_url, "error": str(e)})

    def analyze_sentiment(self, text):
        try:
            sentiment = TextBlob(text).sentiment
            return json.dumps({"text": text, "sentiment": {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}})
        except Exception as e:
            return json.dumps({"text": text, "error": str(e)})

    def setup_client(self):
        try:
            self.client, self.model = choose_API_provider()
        except Exception as e:
            print(f"Error setting up client: {e}")

    def chat(self, system_prompt, prompt):
        def handle_rate_limits(headers):
            retry_after = headers.get('retry-after')
            if retry_after:
                print(f"Rate limit reached. Retrying after {retry_after} seconds...")
                time.sleep(float(retry_after))
            else:
                print("Rate limit reached. Retrying after a default interval...")
                time.sleep(10)

        self.chat_history.append({"role": "user", "content": prompt})

        while True:
            try:
                messages = [{"role": "system", "content": system_prompt}] + self.chat_history
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=4096
                )
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                self.chat_history.append(response_message)

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
                                "content": function_response,
                            }
                        )
                    second_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.chat_history
                    )
                    second_response = second_response.choices[0].message.content
                    self.chat_history.append({"role": "assistant", "content": second_response})

                    third_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.chat_history + [
                            {"role": "system", "content": system_prompt},
                            {"role": "assistant", "content": f"I need to fact check the following information:\n[my_last_response] {second_response} [/my_last_response]\nwith another function call, please wait a moment."},
                            {"role": "user", "content": "Sure, take your time."}
                        ]
                    )
                    third_response_content = third_response.choices[0].message.content
                    self.chat_history.append({"role": "assistant", "content": third_response_content})
                    return third_response_content
                else:
                    return response_message.content
            except APIConnectionError as e:
                print(f"API connection error: {e}. Retrying...")
                time.sleep(1)
            except APIStatusError as e:
                headers = e.response.headers
                handle_rate_limits(headers)
                print(f"API status error: {e}. Retrying...")
            except Exception as e:
                print(f"Error in chat: {e}. Retrying...")
                time.sleep(1)

