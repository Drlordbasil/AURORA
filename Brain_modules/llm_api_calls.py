import json
import os
import subprocess
import time
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
from kivy.clock import Clock
from PyPDF2 import PdfFileReader
from io import BytesIO
from datetime import datetime

def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def test_python_code_on_windows_subprocess(script_raw):
    try:
        with open("temp_script.py", "w") as file:
            file.write(script_raw)
        result = subprocess.run(["python", "temp_script.py"], shell=True, capture_output=True, text=True)
        output = result.stdout
        error = result.stderr
        if error:
            output = json.dumps({"message": f"Script executed with errors: {error}", "datetime": get_current_datetime()})
        else:
            output = json.dumps({"message": f"Script executed successfully: {output}", "datetime": get_current_datetime()})
        return json.dumps({"script": script_raw, "output": output, "error": error})
    except Exception as e:
        error_message = json.dumps({"message": f"Error executing script: {str(e)}", "datetime": get_current_datetime()})
        return json.dumps({"script": script_raw, "error": error_message})

chromadb_client = chromadb.Client()
collection = chromadb_client.create_collection(name="aurora_test_phase")

def time_now():
    return datetime.now().strftime("%H:%M:%S")

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

class WebResearchTool:
    def __init__(self, status_update_callback):
        self.status_update_callback = status_update_callback

    def _update_status(self, message):
        """Update the status through the callback function."""
        self.status_update_callback(message)

    def _initialize_webdriver(self):
        """Initialize and return a Chrome WebDriver with predefined options."""
        options = webdriver.ChromeOptions()
        self._update_status("Initializing Chrome WebDriver...")
        options.add_argument('--disable-gpu')
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver

    def extract_text_from_url(self, url):
        """Extract and return the main text content from a given URL."""
        driver = self._initialize_webdriver()
        self._update_status(f"Extracting text from {url}...")
        try:
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            # Remove unwanted elements (ads, sidebars, etc.)
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'iframe', 'noscript', 'svg']):
                element.extract()
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            if len(text) < 50:  # Arbitrary threshold for minimal content
                raise ValueError("Insufficient content")
            self._update_status(f"Text extracted from {url}: {text[:100]}...")  # Displaying only the first 100 characters
            return text
        except Exception as e:
            self._update_status(f"Error extracting text from {url}: {str(e)}")
            return f"Error extracting text from {url}: {str(e)}"
        finally:
            driver.quit()
            self._update_status("Driver quit successfully.")

    def web_research(self, query):
        """Perform web research based on the given query and return aggregated content."""
        search_engines = ["https://www.google.com", "https://www.bing.com"]
        search_results = []
        self._update_status(f"Performing web research for: {query}")
        for engine in search_engines:
            driver = self._initialize_webdriver()
            try:
                driver.get(engine)
                search_box = driver.find_element(By.NAME, "q")
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)
                self._update_status(f"Searching for {query} using {engine}...")
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                results = soup.select('.tF2Cxc')[:2] if 'google' in engine else soup.select('.b_algo')[:2]
                self._update_status(f"Search results found: {len(results)}")
                for result in results:
                    title_element = result.select_one('.DKV0Md') if 'google' in engine else result.select_one('h2')
                    link_element = result.select_one('a')
                    if title_element and link_element:
                        title = title_element.get_text()
                        link = link_element.get('href')
                        self._update_status(f"Extracting content from {link}...")
                        main_content = self.extract_text_from_url(link)
                        self._update_status(f"Content extracted from {link}: {main_content[:100]}...")  # Displaying only the first 100 characters
                        if "Error extracting text" not in main_content:  # Check if the content extraction was successful
                            search_results.append({
                                "title": title,
                                "link": link,
                                "content": main_content,
                                "datetime": get_current_datetime()
                            })

                            # Fetch additional sub-links for more information
                            driver.get(link)
                            sub_soup = BeautifulSoup(driver.page_source, 'html.parser')
                            sub_links = sub_soup.select('a')[:1]
                            for sub_link in sub_links:
                                sub_url = sub_link.get('href')
                                if sub_url and sub_url.startswith('http'):
                                    sub_content = self.extract_text_from_url(sub_url)
                                    if "Error extracting text" not in sub_content:  # Check if the content extraction was successful
                                        search_results.append({
                                            "title": f"Sublink from {title}",
                                            "link": sub_url,
                                            "content": sub_content,
                                            "datetime": get_current_datetime()
                                        })
                                        self._update_status(f"Sublink content extracted from {sub_url}: {sub_content[:100]}...")
            except Exception as e:
                print(f"Error using {engine}: {e}")
            finally:
                driver.quit()

        # Aggregate and filter the content
        self._update_status(f"Aggregating content for {query}...")
        aggregated_content = []
        for result in search_results:
            sentences = sent_tokenize(result['content'])
            filtered_sentences = [sentence for sentence in sentences if TextBlob(sentence).sentiment.subjectivity < 0.5]
            aggregated_content.extend(filtered_sentences)
            self._update_status(f"Aggregated content from {result['link']}: {len(filtered_sentences)} sentences")
        final_content = ' '.join(aggregated_content)
        if len(final_content.split()) > 1500:
            final_content = ' '.join(final_content.split()[:1500])
            self._update_status(f"Final content aggregated for {query}: {len(final_content.split())} words")
        return json.dumps({"query": query, "results": final_content, "datetime": get_current_datetime()}) if final_content else json.dumps({"query": query, "error": "No results found.", "datetime": get_current_datetime()})

    def extract_text_from_pdf(self, pdf_url):
        self._update_status(f"Extracting text from PDF: {pdf_url}")
        try:
            response = requests.get(pdf_url)
            self._update_status("PDF downloaded successfully.")
            pdf_reader = PdfFileReader(BytesIO(response.content))
            text = ""
            for page_num in range(pdf_reader.numPages):
                text += pdf_reader.getPage(page_num).extract_text()
            self._update_status(f"Text extracted from PDF: {text[:100]}...")  # Displaying only the first 100 characters
            return json.dumps({"pdf_url": pdf_url, "text": text, "datetime": get_current_datetime()})
        except Exception as e:
            self._update_status(f"Error extracting text from PDF: {str(e)}")
            return json.dumps({"pdf_url": pdf_url, "error": str(e), "datetime": get_current_datetime()})

    def analyze_sentiment(self, text):
        try:
            sentiment = TextBlob(text).sentiment
            self._update_status(f"Sentiment analysis for: {text}")
            return json.dumps({"text": text, "sentiment": {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}, "datetime": get_current_datetime()})
        except Exception as e:
            self._update_status(f"Error analyzing sentiment: {str(e)}")
            return json.dumps({"text": text, "error": str(e), "datetime": get_current_datetime()})

class LLM_API_Calls:
    def __init__(self, status_update_callback):
        self.client = None
        self.model = None
        self.setup_client()
        self.image_vision = ImageVision()
        self.web_research_tool = WebResearchTool(status_update_callback)
        self.available_functions = {
            "run_local_command": self.run_local_command,
            "web_research": self.web_research_tool.web_research,
            "analyze_image": self.image_vision.analyze_image,
            "extract_text_from_pdf": self.web_research_tool.extract_text_from_pdf,
            "analyze_sentiment": self.web_research_tool.analyze_sentiment,
            "check_os_default_calendar": self.check_os_default_calendar
        }
        self.chat_history = []
        self.status_update_callback = status_update_callback

    def _update_status(self, message):
        """Update the status through the callback function."""
        Clock.schedule_once(lambda dt: self.status_update_callback(message), 0)

    def check_os_default_calendar(self, date=None, time=None, event_title=None, event_description=None):
        """Check the calendar for today or create a calendar event."""
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
        """Execute a local command on the system to perform tasks such as file manipulation, retrieving system information, or running scripts."""
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

    def setup_client(self):
        try:
            self.client, self.model = choose_API_provider()
            self._update_status("Client setup successful.")
        except Exception as e:
            print(f"Error setting up client: {e}")
            self._update_status(f"Error setting up client: {e}")

    def chat(self, system_prompt, prompt):
        def handle_rate_limits(headers):
            retry_after = headers.get('retry-after')
            self._update_status(f"Rate limit reached. Retrying after {retry_after} seconds...")
            if retry_after:
                print(f"Rate limit reached. Retrying after {retry_after} seconds...")
                time.sleep(float(retry_after))
            else:
                print("Rate limit reached. Retrying after a default interval...")
                self._update_status("Rate limit reached. Retrying after a default interval...")
                time.sleep(30)

        self.chat_history.append({"role": "user", "content": prompt})
        self._update_status("Starting chat process.")
        while True:
            try:
                system_prompt = f"""
                {system_prompt}
                current time: {time.strftime('%H:%M:%S')}
                current date and time: {get_current_datetime()}
                """
                messages = [{"role": "system", "content": system_prompt}] + self.chat_history
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=4096
                )
                response_message = response.choices[0].message
                self._update_status("Chat loop 1/2 completed.")
                tool_calls = response_message.tool_calls
                self.chat_history.append({"role": "assistant", "content": response_message.content})
                self._update_status("Chat loop 2/2 completed.")
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
                        self._update_status(f"Function '{function_name}' executed with args {function_args}: {function_response}")
                    second_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.chat_history
                    )
                    second_response_content = second_response.choices[0].message.content
                    self._update_status("Tool responses processed.")
                    self.chat_history.append({"role": "assistant", "content": second_response_content})
                    return second_response_content
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
