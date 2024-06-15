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

from image_vision import ImageVision

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import chromadb

documents = [
                "",
             ]
chromadb_client = chromadb.Client()
collection = chromadb_client.create_collection(name="aurora_test_phase")

# display current time
def time_now():
    return time.strftime("%H:%M:%S")
time = time_now()
print(time)
tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_local_command",
                    "description": f"""Execute a local command on the system to perform tasks such as file manipulation, retrieving system information, or running scripts.
                    """,
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
                    "description": "Perform a web research query to gather information from online sources. This involves searching for a specific query on Google, extracting relevant content from the top search results, and aggregating the information. For example, you can search for 'Latest AI trends', 'Python 3.12 new features', or 'Benefits of using Docker'.",
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
    """
    Prompts the user to choose an API provider and sets up the client and model accordingly.

    Returns:
        client (object): The API client instance.
        model (str): The model name.
    """
    llm = "Groq" # Change this to the desired API provider: "OpenAI", "ollama", or "Groq"
    
    if llm == "OpenAI":
        api_key = os.environ.get("OPENAI_API_KEY") or input("Enter your OpenAI API key: ").strip()
        model = "gpt-4o"
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
    """
    Class to handle API calls to different LLM providers.
    """

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
    def run_local_command(self, command):
        """
        Execute a local command on the system.
        This function runs a specified command on the local system using the subprocess module.
        It captures the output and error messages from the command execution.
        Parameters:
        - command (str): The specific command to execute. Example commands include:
        - 'dir' to list directory contents
        - 'echo Hello World' to print text
        - 'date /T' to display the current date
        - 'time /T' to show the current time
        - 'python --version' to check the Python version installed
            - 'ipconfig' to display network configuration information
            - 'systeminfo' to show system information
            - 'tasklist' to list running processes
            - 'netstat' to display network connections and port information
            - 'ping <hostname>' to send ICMP echo requests to the specified host
            - 'tracert <hostname>' to trace the route to the specified host
            - 'nslookup <hostname>' to query DNS to resolve the specified hostname
            - 'start chrome' to open the Chrome browser
        Returns:
        - JSON object containing the executed command, its output, and any errors encountered.
        """
        
        try:
            if command.lower().startswith("echo") and ">" in command:
                # Special case: Writing content to a file using 'echo' command
                parts = command.split(">", 1)
                content = parts[0].strip()[5:]  # Remove 'echo' and leading/trailing spaces
                file_path = parts[1].strip()
                with open(file_path, "w") as file:
                    file.write(content)
                output = json.dumps({"message": f"Content written to file: {file_path}"})
                error = ""
            else:
                # Other commands
                if command.lower() == "date":
                    command = "date /T"
                elif command.lower() == "time":
                    command = "time /T"
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

    def extract_text_from_url(self, url):
        """
        Extract text content from a webpage.

        This function uses Selenium to open a webpage, retrieve its HTML content, and extract all text
        within paragraph tags. The function waits for 2 seconds to ensure the page is fully loaded.

        Parameters:
        - url (str): The URL of the webpage to extract text from.

        Returns:
        - The extracted text as a string.
        """
        
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-gpu')
        #options.add_argument('--headless')  # Run in headless mode for efficiency
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        try:
            driver.get(url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            prompt_for_summary = f"Summarize the content of the page.{text}"
            text = self.client.chat.completions.create(model=self.MODEL, messages=[{"role": "user", "content": prompt_for_summary}]).choices[0].message.content
            print("[website_summary]"+text+"[/website_summary]")
            
            return text
        except Exception as e:
            
            return f"Error extracting text from {url}: {str(e)}"
        finally:
            driver.quit()

    def web_research(self, query):
        """
        Perform a web research query.

        This function uses Selenium to perform a Google search for a specified query. It extracts the
        main content from the top search result and its sublinks, filters the content based on sentiment,
        and returns the aggregated content.

        Parameters:
        - query (str): The research query to perform. Example queries include:
          - 'Latest AI trends'
          - 'Python 3.12 new features'
          - 'Benefits of using Docker'

        Returns:
        - JSON object containing the query and the aggregated results.
        """
        
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-gpu')
        #options.add_argument('--headless')  # Run in headless mode for efficiency
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

            # Get top 3 search results
            results = soup.select('.tF2Cxc')[:1]
            for result in results:
                title_element = result.select_one('.DKV0Md')
                link_element = result.select_one('a')

                if title_element and link_element:
                    title = title_element.get_text()
                    link = link_element.get('href')

                    # Extract main page content
                    main_content = self.extract_text_from_url(link)
                    search_results.append({
                        "title": title,
                        "link": link,
                        "content": main_content
                    })

                    # Extract content from 2 sublinks
                    driver.get(link)
                    sub_soup = BeautifulSoup(driver.page_source, 'html.parser')
                    sub_links = sub_soup.select('a')[:1]  # Select first 2 sublinks
                    for sub_link in sub_links:
                        sub_url = sub_link.get('href')
                        if sub_url and sub_url.startswith('http'):
                            sub_content = self.extract_text_from_url(sub_url)
                            search_results.append({
                                "title": f"Sublink from {title}",
                                "link": sub_url,
                                "content": sub_content
                            })

            # Aggregate content and filter based on sentiment
            aggregated_content = []
            for result in search_results:
                sentences = sent_tokenize(result['content'])
                filtered_sentences = []
                for sentence in sentences:
                    sentiment = TextBlob(sentence).sentiment
                    if sentiment.subjectivity < 0.5:  # Filter based on subjectivity
                        filtered_sentences.append(sentence)
                aggregated_content.extend(filtered_sentences)

            # Limit to 1500 words
            final_content = ' '.join(aggregated_content)
            if len(final_content.split()) > 1500:
                final_content = ' '.join(final_content.split()[:1500])

           
            return json.dumps({"query": query, "results": final_content})
        except Exception as e:
            
            return json.dumps({"query": query, "error": str(e)})
        finally:
            driver.quit()

    def extract_text_from_pdf(self, pdf_url):
        """
        Extract text content from a PDF file.

        This function downloads a PDF from the given URL, extracts its text content using pdfminer,
        and returns the extracted text.

        Parameters:
        - pdf_url (str): The URL of the PDF file to extract text from.

        Returns:
        - JSON object containing the PDF URL and the extracted text.
        """
        
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
        """
        Analyze the sentiment of a given text.

        This function uses TextBlob to analyze the sentiment of the provided text. It returns the
        polarity and subjectivity of the sentiment.

        Parameters:
        - text (str): The text to analyze.

        Returns:
        - JSON object containing the text and its sentiment analysis (polarity and subjectivity).
        """

        try:
            sentiment = TextBlob(text).sentiment

            return json.dumps({"text": text, "sentiment": {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}})
        except Exception as e:

            return json.dumps({"text": text, "error": str(e)})
    def setup_client(self):
        """
        Sets up the client and model by prompting the user to choose an API provider.
        """
        try:
            self.client, self.model = choose_API_provider()
        except Exception as e:
            print(f"Error setting up client: {e}")

    def chat(self, system_prompt,prompt):
        """
        Sends a chat prompt to the selected API provider and returns the response.

        Args:
            prompt (str): The chat prompt.

        Returns:
            str: The response from the API.
        """
        try:
            if isinstance(self.client, OpenAI):
    
                messages = [
                    {"role": "user", "content": system_prompt},
                    
                    {"role": "user", "content": prompt}
                            ]

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",  # auto is default, but we'll be explicit
                )
                response_message = response.choices[0].message
                # add response_message to memory
                
                tool_calls = response_message.tool_calls
                # Step 2: check if the model wanted to call a function
                if tool_calls:
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid; be sure to handle errors
                    available_functions = self.available_functions
                    messages.append(response_message)  # extend conversation with assistant's reply
                    # Step 4: send the info for each function call and function response to the model
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_to_call = available_functions[function_name]
                        function_args = json.loads(tool_call.function.arguments)
                        function_response = function_to_call(
                            **function_args
                        )
                            
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )  # extend conversation with function response
                    second_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                    )  # get a new response from the model where it can see the function response
                    second_response = second_response.choices[0].message.content
                    return second_response

            elif self.client == ollama:
                model = OllamaFunctions(model=self.model, format="json")
                model = model.bind_tools(
                    tools=[
                        {
                            "name": "get_current_weather",
                            "description": "Get the current weather in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                                },
                                "required": ["location"],
                            },
                        }
                    ],
                    function_call={"name": "get_current_weather"},
                )

                input_data = {"location": prompt, "unit": "fahrenheit"}
                function_output = model.invoke(json.dumps(input_data))

                summary_prompt = f"""
                Please provide a summary of the usage of this code, including:
                - The purpose of the get_current_weather function
                - The input data provided to the get_current_weather function: {input_data}
                - The output returned by the get_current_weather function: {function_output.content}
                - the original prompt: {prompt}
                """

                llm = ollama.chat(model=self.model, messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": summary_prompt}
                                                              ])
                summary_output = llm["message"]["content"]

                return summary_output

            elif isinstance(self.client, Groq):
                try:


                    # Construct the messages and tools
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]


                    # Make the initial API call
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=4096
                    )

                    response_message = response.choices[0].message
                    tool_calls = response_message.tool_calls

                    # Check if the model requested to call a function
                    if tool_calls:
                        available_functions = self.available_functions
                        messages.append(response_message)

                        # Execute the requested functions and append the responses to the conversation
                        for tool_call in tool_calls:
                            function_name = tool_call.function.name
                            function_to_call = available_functions[function_name]
                            function_args = json.loads(tool_call.function.arguments)
                            function_response = function_to_call(
                                **function_args

                            )
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": function_response,
                                }
                            )

                        # Make a second API call with the updated conversation
                        second_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages
                        )
                        return second_response.choices[0].message.content
                    else:
                        return response_message.content

                except Exception as e:
                    print(f"Error in Groq tool usage: {e}")
                    return None

        except APIConnectionError as e:
            print(f"API connection error: {e}")
            return None
        except APIStatusError as e:
            print(f"API status error: {e}")
            return None
        except Exception as e:
            print(f"Error in chat: {e}")
            return None

# # Example usage: A chatbot loop
# llm_api = LLM_API_Calls()

# def chat_loop():
#     print("Welcome to the chatbot. Type 'exit' to end the chat.")
#     system_prompt = f"You are a multi-use function calling LLM. current time is {time}"
    
#     while True:
#         try:
#             prompt = input("You: ")
#             if prompt.lower() == 'exit':
#                 print("Goodbye!")
#                 break
        
#             response = llm_api.chat(system_prompt=system_prompt, prompt=prompt)
        
#             if response:
#                 print("assistant:", response)
#                 prompt = response # Using the response as the next prompt, if applicable
#             else:
#                 print("Failed to generate a response.")

#         except KeyboardInterrupt:
#             print("Goodbye!")
#             break
#         except Exception as e:
#             print(f"An error occurred: {e}")
#             break
# # Start the chatbot loop
# chat_loop()
