import json
import subprocess
import os
import time
import requests
from groq import Groq
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from image_vision import ImageVision
from pdfminer.high_level import extract_text
from kivy.clock import Clock

class FunctionCalling:
    MODEL = 'llama3-70b-8192'

    def __init__(self, api_key, status_update_callback):
        self.client = Groq(api_key=api_key)
        self.image_vision = ImageVision()
        self.status_update_callback = status_update_callback

    def _update_status(self, message):
        """
        Update the GUI with the current status message.

        Args:
            message (str): The status message to be displayed on the GUI.
        """
        Clock.schedule_once(lambda dt: self.status_update_callback(message), 0)

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

        Returns:
        - JSON object containing the executed command, its output, and any errors encountered.
        """
        self._update_status(f"Running local command: {command}")
        try:
            if command.lower() == "date":
                command = "date /T"
            elif command.lower() == "time":
                command = "time /T"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout
            error = result.stderr
            self._update_status(f"Completed local command: {command}")
            return json.dumps({"command": command, "output": output, "error": error})
        except Exception as e:
            self._update_status(f"Error running local command: {command}")
            return json.dumps({"command": command, "error": str(e)})

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
        self._update_status(f"Extracting text from URL: {url}")
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
            self._update_status(f"Completed text extraction from URL: {url}")
            return text
        except Exception as e:
            self._update_status(f"Error extracting text from URL: {url}")
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
        self._update_status(f"Performing web research for query: {query}")
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

            self._update_status(f"Completed web research for query: {query}")
            return json.dumps({"query": query, "results": final_content})
        except Exception as e:
            self._update_status(f"Error performing web research for query: {query}")
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
        self._update_status(f"Extracting text from PDF: {pdf_url}")
        try:
            response = requests.get(pdf_url)
            with open("temp.pdf", "wb") as file:
                file.write(response.content)
            text = extract_text("temp.pdf")
            os.remove("temp.pdf")
            self._update_status(f"Completed text extraction from PDF: {pdf_url}")
            return json.dumps({"pdf_url": pdf_url, "text": text})
        except Exception as e:
            self._update_status(f"Error extracting text from PDF: {pdf_url}")
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
        self._update_status("Analyzing sentiment")
        try:
            sentiment = TextBlob(text).sentiment
            self._update_status("Sentiment analysis completed")
            return json.dumps({"text": text, "sentiment": {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}})
        except Exception as e:
            self._update_status("Error analyzing sentiment")
            return json.dumps({"text": text, "error": str(e)})

    def run_conversation(self, user_prompt):
        self._update_status("Running conversation")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a function calling LLM that uses the data extracted from local commands "
                    "and web research functions to provide detailed responses to the user. You are "
                    "designed to assist with running system commands and performing web research to "
                    "gather relevant information. You are Aurora's assistant ONLY. Aurora is the main "
                    "chatbot and you are its assistant that can perform specific tasks. You can run "
                    "local commands, perform web research, and analyze images to provide detailed "
                    "responses to the user. You can also interact with other tools and services to "
                    "enhance the user experience. Here are the specific tasks you can perform:\n"
                    "- 'run_local_command': Execute a local command on the system. Example commands "
                    "include: 'dir' to list directory contents, 'echo Hello World' to print text, "
                    "'date /T' to display the current date, 'time /T' to show the current time, "
                    "'python --version' to check the Python version installed.\n"
                    "- 'web_research': Perform a web research query to gather information from online "
                    "sources. Example queries include: 'Latest AI trends', 'Python 3.12 new features', "
                    "'Benefits of using Docker'.\n"
                    "- 'analyze_image': Analyze an image from a provided URL and generate a description "
                    "of the image's content.\n"
                    "- 'extract_text_from_pdf': Extract text content from a PDF file. Provide the URL "
                    "of the PDF file.\n"
                    "- 'analyze_sentiment': Analyze the sentiment of a given text, providing polarity "
                    "and subjectivity scores."
                )
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
                    "description": "Execute a local command on the system to perform tasks such as file manipulation, retrieving system information, or running scripts. Example commands include: 'dir' to list directory contents, 'echo Hello World' to print text, 'date /T' to display the current date, 'time /T' to show the current time, and 'python --version' to check the Python version installed. ONLY use valid command for what user wants IE: if user asks for chrome browser, open it with command 'chrome' not 'dir' or 'echo Hello World' etc.",
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
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096
        )
        time.sleep(10)
        self._update_status("first response received in function_calling loop")
        response_message = response.choices[0].message
        time.sleep(10)
        tool_calls = response_message.tool_calls
        time.sleep(10)
        self._update_status("tool calls received in function_calling loop")
        if tool_calls:
            available_functions = {
                "run_local_command": self.run_local_command,
                "web_research": self.web_research,
                "analyze_image": self.image_vision.analyze_image,
                "extract_text_from_pdf": self.extract_text_from_pdf,
                "analyze_sentiment": self.analyze_sentiment,
            }
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                retry_count = 0
                max_retries = 3
                self._update_status(f"Calling function: {function_name}")
                while retry_count < max_retries:
                    try:
                        function_response = function_to_call(**function_args)
                        self._update_status(f"Function {function_name} completed")
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            function_response = json.dumps({"error": f"Failed after {max_retries} retries: {str(e)}"})
                        else:
                            function_response = json.dumps({"error": f"Attempt {retry_count}/{max_retries} failed: {str(e)}. Retrying..."})
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
        api_key = input("Enter your GROQ API key: ")
    else:
        from brain import Brain  # Assuming we have a Brain instance to     pass the add_to_memory method
        brain_instance = Brain(api_key)
        fc = FunctionCalling(api_key)
        
        # Test prompts for all functions
        user_prompts = [
            "analyze this image:https://i.gyazo.com/3fec0307e8c73b23d3871d113d63647d.png",
            "Show the current Python version.",
            "Look up the benefits of using Docker.",
            "Extract text from this PDF: https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "Analyze the sentiment of this text: I am so happy with the results!"
        ]
        for prompt in user_prompts:
            print(f"Testing run_conversation with prompt: {prompt}")
            print(fc.run_conversation(prompt))
            time.sleep(10)  # Sleep for 10 seconds between tests
