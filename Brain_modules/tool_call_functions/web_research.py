import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import trafilatura
from selenium.common.exceptions import WebDriverException, NoSuchElementException

class WebResearchTool:
    def __init__(self, chat_function, max_tokens=4000):
        self.chat_function = chat_function
        self.max_tokens = max_tokens

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
        except Exception:
            return None

    def summarize_content(self, content, query):
        try:
            system_prompt = "You are a highly skilled researcher and summarizer. Your task is to create a concise, informative summary of the provided content, focusing on the most relevant information related to the given query."
            user_prompt = f"Query: {query}\n\nContent to summarize: {content}\n\nPlease provide a concise summary (around 150 words) that captures the most important and relevant information related to the query."
            
            max_content_tokens = self.max_tokens - len(system_prompt.split()) - len(user_prompt.split()) - 200
            if len(content.split()) > max_content_tokens:
                content = ' '.join(content.split()[:max_content_tokens]) + "..."
                user_prompt = f"Query: {query}\n\nContent to summarize (truncated): {content}\n\nPlease provide a concise summary (around 150 words) that captures the most important and relevant information related to the query."

            summary = self.chat_function(system_prompt, user_prompt)
            return summary.strip()
        except Exception:
            return content[:1000] + "..." if len(content) > 1000 else content

    def web_research(self, query):
        search_engines = [
            ("https://www.google.com", 'input[name="q"]'),
            ("https://www.bing.com", 'input[name="q"]'),
            ("https://search.brave.com", 'input[name="q"]')
        ]
        search_results = []

        for engine, search_box_selector in search_engines:
            try:
                driver = self._initialize_webdriver()
                driver.get(engine)
                search_box = driver.find_element(By.CSS_SELECTOR, search_box_selector)
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)
                
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                results = soup.select('a')[:5]  # Top 5 results

                for result in results:
                    link = result.get('href')
                    if link and link.startswith('http'):
                        content = self.extract_text_from_url(link)
                        if content:
                            search_results.append({
                                "title": result.get_text(),
                                "link": link,
                                "content": content,
                            })
            except (WebDriverException, NoSuchElementException):
                continue
            finally:
                driver.quit()

        if not search_results:
            return f"No results found for the query: {query}"

        aggregated_content = "\n\n".join([result['content'] for result in search_results])
        summary = self.summarize_content(aggregated_content, query)

        return summary if summary else f"Unable to summarize results for the query: {query}"