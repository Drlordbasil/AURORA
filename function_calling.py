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
from nltk.tokenize import sent_tokenize
from textblob import TextBlob

class FunctionCalling:
    MODEL = 'llama3-70b-8192'

    def __init__(self, api_key, memory_handler):
        self.client = Groq(api_key=api_key)
        self.memory_handler = memory_handler  # Initialize with memory handler

    def run_local_command(self, command):
        if command.lower() == "date":
            command = "date /T"
        elif command.lower() == "time":
            command = "time /T"
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout
            error = result.stderr
            
            # Add to memory
            self.memory_handler(f"Command: {command}\nOutput: {output}\nError: {error}")

            return json.dumps({"command": command, "output": output, "error": error})
        except Exception as e:
            error_message = str(e)
            
            # Add to memory
            self.memory_handler(f"Command: {command}\nError: {error_message}")

            return json.dumps({"command": command, "error": error_message})

    def extract_text_from_url(self, url):
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-gpu')
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        try:
            driver.get(url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            return text
        except Exception as e:
            return f"Error extracting text from {url}: {str(e)}"
        finally:
            driver.quit()

    def web_research(self, query):
        options = webdriver.ChromeOptions()
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

            # Get top 3 search results
            results = soup.select('.tF2Cxc')[:3]
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
                    
                    # Extract content from 3 sublinks
                    driver.get(link)
                    sub_soup = BeautifulSoup(driver.page_source, 'html.parser')
                    sub_links = sub_soup.select('a')[:3]  # Select first 3 sublinks
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

            # Add to memory
            self.memory_handler(f"Query: {query}\nResults: {final_content}")

            return json.dumps({"query": query, "results": final_content})
        except Exception as e:
            error_message = str(e)
            
            # Add to memory
            self.memory_handler(f"Query: {query}\nError: {error_message}")

            return json.dumps({"query": query, "error": error_message})
        finally:
            driver.quit()

    def run_conversation(self, user_prompt):
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
                    "description": "Execute a local command on the system to perform tasks such as file manipulation, retrieving system information, or running scripts. Example commands include: 'dir' to list directory contents, 'echo Hello World' to print text, 'date /T' to display the current date, 'time /T' to show the current time, and 'python --version' to check the Python version installed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The specific command to execute on the local system. Examples: 'dir', 'echo Hello World', 'date /T', 'time /T', 'python --version'.",
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
                                "description": "The research query to perform. Example queries include: 'Latest AI trends', 'Python 3.12 new features', 'Benefits of using Docker', 'How to set up a virtual environment in Python', 'Top 10 programming languages in 2024'.",
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
        # Assuming we have a Brain instance to pass the add_to_memory method
        from brain import Brain
        brain_instance = Brain(api_key)
        fc = FunctionCalling(api_key, brain_instance.add_to_memory)
        
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
