import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from kivy.clock import Clock
from datetime import datetime
import trafilatura

class WebResearchTool:
    def __init__(self, status_update_callback, chat_function, max_tokens=4000):
        self.status_update_callback = status_update_callback
        self.chat_function = chat_function
        self.max_tokens = max_tokens

    def _update_status(self, message):
        self.status_update_callback(message)

    def _initialize_webdriver(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-extensions')
        service = ChromeService(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def extract_text_from_url(self, url):
        try:
            self._update_status(f"Extracting content from {url}")
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                if text and len(text) >= 50:
                    return text
            
            # Fallback to Selenium if trafilatura fails
            driver = self._initialize_webdriver()
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'aside']):
                element.decompose()
            text = ' '.join(p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 20)
            driver.quit()
            return text if len(text) >= 50 else None
        except Exception as e:
            self._update_status(f"Error extracting text from {url}: {str(e)}")
            return None

    def summarize_content(self, content, query):
        try:
            self._update_status("Generating summary")
            system_prompt = "You are a highly skilled researcher and summarizer. Your task is to create a concise, informative summary of the provided content, focusing on the most relevant information related to the given query."
            user_prompt = f"Query: {query}\n\nContent to summarize: {content}\n\nPlease provide a concise summary (around 150 words) that captures the most important and relevant information related to the query."
            
            # Limit the content to ensure we don't exceed token limits
            max_content_tokens = self.max_tokens - len(system_prompt.split()) - len(user_prompt.split()) - 200  # 200 tokens buffer
            if len(content.split()) > max_content_tokens:
                content = ' '.join(content.split()[:max_content_tokens]) + "..."
                user_prompt = f"Query: {query}\n\nContent to summarize (truncated): {content}\n\nPlease provide a concise summary (around 150 words) that captures the most important and relevant information related to the query."

            summary = self.chat_function(system_prompt, user_prompt)
            return summary.strip()
        except Exception as e:
            self._update_status(f"Error in summarization: {str(e)}")
            return content[:1000] + "..." if len(content) > 1000 else content

    def web_research(self, query):
        search_engines = [
            ("https://www.google.com", '.g'),
            ("https://www.bing.com", '.b_algo'),
            ("https://search.brave.com", '.snippet')
        ]
        search_results = []
        self._update_status(f"Performing web research for: {query}")

        for engine, result_selector in search_engines:
            driver = self._initialize_webdriver()
            try:
                driver.get(engine)
                search_box = driver.find_element(By.NAME, "q")
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)
                self._update_status(f"Searching {engine} for {query}")
                
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                results = soup.select(result_selector)[:5]  # Top 5 results

                for result in results:
                    link_element = result.select_one('a')
                    if link_element and link_element.get('href'):
                        link = link_element['href']
                        if link.startswith('http'):
                            content = self.extract_text_from_url(link)
                            if content:
                                search_results.append({
                                    "title": link_element.get_text(),
                                    "link": link,
                                    "content": content,
                                    "datetime": datetime.now().isoformat()
                                })
                                self._update_status(f"Content extracted from {link}")

            except Exception as e:
                self._update_status(f"Error using {engine}: {str(e)}")
            finally:
                driver.quit()

        # Aggregate content
        self._update_status("Aggregating content")
        aggregated_content = "\n\n".join([result['content'] for result in search_results])

        # Summarize content
        summary = self.summarize_content(aggregated_content, query)

        self._update_status(f"Summary generated: {len(summary.split())} words")
        return json.dumps({
            "query": query,
            "summary": summary,
            "datetime": datetime.now().isoformat()
        }) if summary else json.dumps({
            "query": query,
            "error": "No results found.",
            "datetime": datetime.now().isoformat()
        })