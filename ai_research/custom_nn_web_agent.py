import time
import numpy as np
import os
import pickle
import logging
from typing import List, Dict, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from datetime import datetime
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global list to capture errors for the summary
error_log = []

# Simple sentiment analysis dictionary (AFINN)
SENTIMENT_DICT = {
    "good": 3, "great": 5, "excellent": 7, "positive": 4, "fortunate": 2, "correct": 2, "superior": 5,
    "bad": -3, "terrible": -5, "poor": -2, "negative": -4, "unfortunate": -2, "wrong": -3, "inferior": -5
}

class ReshapeSuperhighway:
    """
    A utility class to handle reshaping operations.
    """
    @staticmethod
    def flatten(array: np.ndarray) -> np.ndarray:
        """
        Flatten the given array.

        Parameters:
        array (np.ndarray): The array to flatten.

        Returns:
        np.ndarray: The flattened array.
        """
        return array.flatten()

    @staticmethod
    def reshape_to_original(array: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reshape the array to its original shape.

        Parameters:
        array (np.ndarray): The array to reshape.
        original_shape (Tuple[int, ...]): The original shape to reshape the array into.

        Returns:
        np.ndarray: The reshaped array.
        """
        try:
            return array.reshape(original_shape)
        except ValueError as e:
            error_message = f"Error reshaping array to shape {original_shape}: {str(e)}"
            logging.error(error_message)
            error_log.append(error_message)
            raise

class CustomConvLayer:
    def __init__(self, num_filters, filter_size):
        """
        Initialize the convolutional layer.

        Parameters:
        num_filters (int): The number of filters.
        filter_size (int): The size of each filter.
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9

    def iterate_regions(self, image):
        """
        Generate all possible 3x3 image regions using valid padding.

        Parameters:
        image (np.ndarray): The input image.

        Yields:
        Tuple[np.ndarray, int, int]: A tuple containing the image region, its row index, and column index.
        """
        h, w = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield region, i, j

    def forward(self, input):
        """
        Perform the forward pass of the convolutional layer.

        Parameters:
        input (np.ndarray): The input image.

        Returns:
        np.ndarray: The output of the convolutional layer.
        """
        self.last_input = input
        h, w = input.shape
        self.last_output_shape = (h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters)
        output = np.zeros(self.last_output_shape)
        for region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))
        return output

    def backward(self, d_L_d_out, learning_rate):
        """
        Perform the backward pass of the convolutional layer.

        Parameters:
        d_L_d_out (np.ndarray): The gradient of the loss with respect to the output.
        learning_rate (float): The learning rate.

        Returns:
        np.ndarray: The gradient of the loss with respect to the filters.
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        for region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * region
        self.filters -= learning_rate * d_L_d_filters
        return d_L_d_filters

class CustomDenseLayer:
    def __init__(self, input_len, output_len):
        """
        Initialize the dense layer.

        Parameters:
        input_len (int): The length of the input.
        output_len (int): The length of the output.
        """
        self.weights = np.random.randn(input_len, output_len) / np.sqrt(input_len)
        self.biases = np.zeros(output_len)

    def forward(self, input):
        """
        Perform the forward pass of the dense layer.

        Parameters:
        input (np.ndarray): The input array.

        Returns:
        np.ndarray: The output of the dense layer.
        """
        self.last_input_shape = input.shape
        input = ReshapeSuperhighway.flatten(input)
        self.last_input = input
        self.last_output = np.dot(input, self.weights) + self.biases
        return self.last_output

    def backward(self, d_L_d_out, learning_rate):
        """
        Perform the backward pass of the dense layer.

        Parameters:
        d_L_d_out (np.ndarray): The gradient of the loss with respect to the output.
        learning_rate (float): The learning rate.

        Returns:
        np.ndarray: The gradient of the loss with respect to the input.
        """
        d_L_d_weights = np.dot(self.last_input[:, None], d_L_d_out[None, :])
        d_L_d_biases = d_L_d_out
        d_L_d_input = np.dot(d_L_d_out, self.weights.T)
        self.weights -= learning_rate * d_L_d_weights
        self.biases -= learning_rate * d_L_d_biases
        return ReshapeSuperhighway.reshape_to_original(d_L_d_input, self.last_input_shape)

class CustomRNNLayer:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the RNN layer.

        Parameters:
        input_size (int): The size of the input.
        hidden_size (int): The size of the hidden state.
        output_size (int): The size of the output.
        """
        self.hidden_size = hidden_size
        self.Whh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.Wxh = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
        self.Why = np.random.randn(output_size, hidden_size) / np.sqrt(hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        Perform the forward pass of the RNN layer.

        Parameters:
        inputs (List[np.ndarray]): A list of input arrays for each time step.

        Returns:
        np.ndarray: The output of the RNN layer.
        """
        self.inputs = inputs
        self.hs = {}
        self.hs[-1] = np.zeros((self.hidden_size, 1))
        for t in range(len(inputs)):
            self.hs[t] = np.tanh(np.dot(self.Wxh, inputs[t]) + np.dot(self.Whh, self.hs[t-1]) + self.bh)
        self.output = np.dot(self.Why, self.hs[len(inputs)-1]) + self.by
        return self.output

    def backward(self, d_L_d_out, learning_rate):
        """
        Perform the backward pass of the RNN layer.

        Parameters:
        d_L_d_out (np.ndarray): The gradient of the loss with respect to the output.
        learning_rate (float): The learning rate.
        """
        d_Why = np.dot(d_L_d_out, self.hs[len(self.inputs)-1].T)
        d_by = d_L_d_out
        d_h = np.dot(self.Why.T, d_L_d_out)
        for t in reversed(range(len(self.inputs))):
            temp = (1 - self.hs[t] ** 2) * d_h
            d_bh = temp
            d_Wxh = np.dot(temp, self.inputs[t].T)
            d_Whh = np.dot(temp, self.hs[t-1].T)
            d_h = np.dot(self.Whh.T, temp)
            self.Whh -= learning_rate * d_Whh
            self.Wxh -= learning_rate * d_Wxh
            self.bh -= learning_rate * d_bh
        self.Why -= learning_rate * d_Why
        self.by -= learning_rate * d_by

class Seq2SeqRNN:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize the Seq2Seq RNN.

        Parameters:
        input_size (int): The size of the input.
        hidden_size (int): The size of the hidden state.
        output_size (int): The size of the output.
        num_layers (int): The number of layers.
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = CustomRNNLayer(input_size, hidden_size, hidden_size)
        self.decoder = CustomRNNLayer(hidden_size, hidden_size, output_size)

    def forward(self, input_seq):
        """
        Perform the forward pass of the Seq2Seq RNN.

        Parameters:
        input_seq (List[np.ndarray]): A list of input arrays for each time step.

        Returns:
        List[np.ndarray]: A list of output arrays for each time step.
        """
        encoder_outputs = self.encoder.forward(input_seq)
        decoder_input = encoder_outputs[-1]  # Use the last encoder output as the initial decoder input
        decoder_outputs = [decoder_input]

        for _ in range(len(input_seq) - 1):  # Generate a sequence of the same length as input
            decoder_input = self.decoder.forward([decoder_input])
            decoder_outputs.append(decoder_input)

        return decoder_outputs

    def backward(self, d_L_d_out, learning_rate):
        """
        Perform the backward pass of the Seq2Seq RNN.

        Parameters:
        d_L_d_out (np.ndarray): The gradient of the loss with respect to the output.
        learning_rate (float): The learning rate.
        """
        self.decoder.backward(d_L_d_out, learning_rate)
        self.encoder.backward(d_L_d_out, learning_rate)

class MultimodalAgent:
    def __init__(self, input_shape: Tuple[int, int], num_classes: int, hidden_size: int, output_size: int):
        """
        Initialize the Multimodal Agent.

        Parameters:
        input_shape (Tuple[int, int]): The shape of the input.
        num_classes (int): The number of classes.
        hidden_size (int): The size of the hidden state.
        output_size (int): The size of the output.
        """
        self.cnn = CustomConvLayer(8, 3)
        self.dense = CustomDenseLayer((input_shape[0] - 2) * (input_shape[1] - 2) * 8, num_classes)
        self.rnn = CustomRNNLayer(num_classes, hidden_size, output_size)

    def forward(self, input: np.ndarray, text_input: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the forward pass of the Multimodal Agent.

        Parameters:
        input (np.ndarray): The input array.
        text_input (List[np.ndarray]): A list of input arrays for each time step.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the action output and the text output.
        """
        out = self.cnn.forward(input)
        out_flat = ReshapeSuperhighway.flatten(out)  # Flatten the CNN output before feeding into the dense layer
        action_output = self.dense.forward(out_flat)
        text_output = self.rnn.forward(text_input)
        return action_output, text_output

    def backward(self, d_L_d_action: np.ndarray, d_L_d_text: np.ndarray, learning_rate: float):
        """
        Perform the backward pass of the Multimodal Agent.

        Parameters:
        d_L_d_action (np.ndarray): The gradient of the loss with respect to the action output.
        d_L_d_text (np.ndarray): The gradient of the loss with respect to the text output.
        learning_rate (float): The learning rate.
        """
        d_L_d_action_flat = ReshapeSuperhighway.flatten(d_L_d_action)  # Flatten the gradient to match the dense layer input
        d_L_d_action_input = self.dense.backward(d_L_d_action_flat, learning_rate)
        d_L_d_action_reshaped = ReshapeSuperhighway.reshape_to_original(d_L_d_action_input, self.cnn.last_output_shape)  # Reshape to match CNN output
        self.cnn.backward(d_L_d_action_reshaped, learning_rate)
        self.rnn.backward(d_L_d_text, learning_rate)

class UnifiedModel:
    def __init__(self, input_shape: Tuple[int, int], num_classes: int, hidden_size: int, output_size: int):
        """
        Initialize the Unified Model.

        Parameters:
        input_shape (Tuple[int, int]): The shape of the input.
        num_classes (int): The number of classes.
        hidden_size (int): The size of the hidden state.
        output_size (int): The size of the output.
        """
        self.multimodal = MultimodalAgent(input_shape, num_classes, hidden_size, output_size)
        self.seq2seq = Seq2SeqRNN(num_classes, hidden_size, output_size)

    def forward(self, input: np.ndarray, text_input: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the forward pass of the Unified Model.

        Parameters:
        input (np.ndarray): The input array.
        text_input (List[np.ndarray]): A list of input arrays for each time step.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the action output, the text output, and the generated text sequence.
        """
        action_output, text_output = self.multimodal.forward(input, text_input)
        generated_text_seq = self.seq2seq.forward(text_input)
        return action_output, text_output, generated_text_seq

    def backward(self, d_L_d_action: np.ndarray, d_L_d_text: np.ndarray, d_L_d_gen_text: np.ndarray, learning_rate: float):
        """
        Perform the backward pass of the Unified Model.

        Parameters:
        d_L_d_action (np.ndarray): The gradient of the loss with respect to the action output.
        d_L_d_text (np.ndarray): The gradient of the loss with respect to the text output.
        d_L_d_gen_text (np.ndarray): The gradient of the loss with respect to the generated text sequence.
        learning_rate (float): The learning rate.
        """
        self.multimodal.backward(d_L_d_action, d_L_d_text, learning_rate)
        self.seq2seq.backward(d_L_d_gen_text, learning_rate)

    def save_model(self, filepath: str = 'unified_model.pkl'):
        """
        Save the Unified Model to a file.

        Parameters:
        filepath (str): The path to the file where the model will be saved.
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logging.info("Unified model saved successfully.")
        except (OSError, IOError) as e:
            error_message = f"Error saving unified model to {filepath}: {str(e)}"
            logging.error(error_message)
            error_log.append(error_message)

    @staticmethod
    def load_model(filepath: str = 'unified_model.pkl') -> 'UnifiedModel':
        """
        Load the Unified Model from a file.

        Parameters:
        filepath (str): The path to the file where the model is saved.

        Returns:
        UnifiedModel: The loaded model.
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    logging.info("Unified model loaded successfully.")
                    return pickle.load(f)
            else:
                logging.warning(f"File {filepath} not found. Creating a new unified model.")
                return UnifiedModel((128, 128), 1, 128, 1)  # Adjust the input shape and other parameters as needed
        except (OSError, IOError, pickle.UnpicklingError) as e:
            error_message = f"Error loading unified model from {filepath}: {str(e)}"
            logging.error(error_message)
            error_log.append(error_message)
            return UnifiedModel((128, 128), 1, 128, 1)  # Return a new model in case of error

def calculate_sentiment_score(text: str) -> int:
    """
    Calculate the sentiment score of the text.

    Parameters:
    text (str): The text to analyze.

    Returns:
    int: The sentiment score.
    """
    try:
        words = text.split()
        sentiment = sum(SENTIMENT_DICT.get(word.lower(), 0) for word in words)
        return sentiment
    except Exception as e:
        error_message = f"Error calculating sentiment score: {str(e)}"
        logging.error(error_message)
        error_log.append(error_message)
        return 0

def calculate_total_reward(content_length: int, job_listings_count: int, sentiment_score: int) -> float:
    """
    Calculate the total reward based on the content length, number of job listings, and sentiment score.

    Parameters:
    content_length (int): The length of the content.
    job_listings_count (int): The number of job listings.
    sentiment_score (int): The sentiment score.

    Returns:
    float: The total reward.
    """
    try:
        return content_length * 0.05 + job_listings_count * 10 + sentiment_score * 2
    except Exception as e:
        error_message = f"Error calculating total reward: {str(e)}"
        logging.error(error_message)
        error_log.append(error_message)
        return 0.0

def select_next_url_to_scrape(urls: List[str], model: UnifiedModel) -> str:
    """
    Select the next URL to scrape using the RL model.

    Parameters:
    urls (List[str]): A list of URLs to choose from.
    model (UnifiedModel): The RL model to use for selecting the URL.

    Returns:
    str: The selected URL.
    """
    try:
        url_scores = []
        for url in urls:
            action_output, _, _ = model.forward(np.random.rand(128, 128), [np.random.rand(1, 1)])  # Dummy input
            url_scores.append(action_output[0])
        return urls[np.argmax(url_scores)]
    except Exception as e:
        error_message = f"Error selecting next URL to scrape: {str(e)}"
        logging.error(error_message)
        error_log.append(error_message)
        return urls[0] if urls else ""

def perform_web_search(query: str, search_engine: str = 'google') -> List[str]:
    """
    Perform a web search using the specified search engine.

    Parameters:
    query (str): The search query.
    search_engine (str): The search engine to use ('google' or 'bing').

    Returns:
    List[str]: A list of URLs from the search results.
    """
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 10)
    urls = []

    logging.debug("Starting search on %s with query: %s", search_engine, query)

    try:
        if search_engine == 'google':
            driver.get('https://www.google.com')
            search_box = wait.until(EC.presence_of_element_located((By.NAME, 'q')))
        elif search_engine == 'bing':
            driver.get('https://www.bing.com')
            search_box = wait.until(EC.presence_of_element_located((By.NAME, 'q')))
        else:
            error_message = f"Unknown search engine: {search_engine}"
            logging.error(error_message)
            error_log.append(error_message)
            return []

        search_box.send_keys(query)
        search_box.submit()
        wait.until(EC.presence_of_element_located((By.ID, 'search' if search_engine == 'google' else 'b_results')))

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        search_results = soup.find_all('a', href=True)

        for result in search_results:
            url = result['href']
            if 'indeed.com' not in url and 'javascript:void(0);' not in url and url.startswith("http"):
                urls.append(url)
            if len(urls) == 3:
                break

        logging.debug("Found URLs: %s", urls)

    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        error_message = f"An error occurred while performing the search: {str(e)}"
        logging.error(error_message)
        error_log.append(error_message)
    finally:
        driver.quit()

    return urls

def extract_job_listings_from_url(url: str, query: str) -> Tuple[str, List[Dict[str, str]], str, str]:
    """
    Extract job listings from the given URL.

    Parameters:
    url (str): The URL to scrape.
    query (str): The search query to look for in the content.

    Returns:
    Tuple[str, List[Dict[str, str]], str, str]: A tuple containing the content, job listings, relevant content, and timestamp.
    """
    job_listings = []
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 10)

    logging.debug("Scraping URL: %s", url)

    try:
        driver.get(url)
        time.sleep(3)  # Wait for the page to load

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        content = soup.text
        timestamp = datetime.now().isoformat()  # Add timestamp to the content
        # Example selectors, update these based on actual HTML structure of the target websites
        job_cards = soup.find_all('div', class_='job_card')

        for job_card in job_cards:
            title_element = job_card.find('h2', class_='job_title')
            company_element = job_card.find('span', class_='company_name')
            location_element = job_card.find('div', class_='job_location')
            url_element = job_card.find('a', href=True)

            title = title_element.text.strip() if title_element else 'N/A'
            company = company_element.text.strip() if company_element else 'N/A'
            location = location_element.text.strip() if location_element else 'N/A'
            job_url = url_element['href'] if url_element else 'N/A'

            job_listings.append({
                "title": title,
                "company": company,
                "location": location,
                "url": job_url,
                "timestamp": timestamp
            })

        # Extract relevant content close to the search query
        relevant_content = " ".join([elem.text for elem in soup.find_all(['h1', 'h2', 'p']) if query.lower() in elem.text.lower()])

    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        error_message = f"An error occurred while scraping URL {url}: {str(e)}"
        logging.error(error_message)
        error_log.append(error_message)
        content = ""
        relevant_content = ""
        timestamp = ""
    finally:
        driver.quit()

    return content, job_listings, relevant_content, timestamp

def run_rl_iteration(search_query: str, all_urls: List[str], model: UnifiedModel, iteration: int) -> float:
    """
    Run a single iteration of the RL process.

    Parameters:
    search_query (str): The search query to use.
    all_urls (List[str]): A list of URLs to choose from.
    model (UnifiedModel): The RL model to use.
    iteration (int): The current iteration number.

    Returns:
    float: The reward for the current iteration.
    """
    print(f"\n=========== Iteration {iteration + 1} ===========\n")

    selected_url = select_next_url_to_scrape(all_urls, model)
    print(f"Selected URL: {selected_url}")

    content, job_listings, relevant_content, timestamp = extract_job_listings_from_url(selected_url, search_query)
    print(f"Scraped Content Length: {len(content)}")
    print(f"Number of Job Listings Found: {len(job_listings)}")
    print(f"Relevant Content: {relevant_content}")

    sentiment_score = calculate_sentiment_score(relevant_content)
    print(f"Sentiment Score: {sentiment_score}")

    reward = calculate_total_reward(len(content), len(job_listings), sentiment_score)
    print(f"Reward for this iteration: {reward}")

    text_input = [np.random.rand(1, 1) for _ in range(len(relevant_content.split()))]
    action_output, text_output, generated_text_seq = model.forward(np.random.rand(128, 128), text_input)
    generated_text = ' '.join([str(output) for output in generated_text_seq])
    print(f"Generated Text: {generated_text}")

    model.backward(np.array([reward]), text_output, generated_text_seq, 0.001)  # Using a learning rate of 0.001

    # Save model after each iteration
    model.save_model()

    print(f"\nAnalyzed Text: {relevant_content}")
    print(f"Sentiment Score: {sentiment_score}")
    print(f"Action Output: {action_output}")
    print(f"Text Output: {text_output}")
    print(f"Generated Text: {generated_text}")

    return reward

def main():
    """
    Main function to run the entire RL process.
    """
    search_queries = [
        "Software Engineer Remote",
        "Data Scientist Jobs",
        "Machine Learning Engineer",
        "AI Researcher Positions",
        "Deep Learning Developer",
        "NLP Specialist Jobs",
        "Computer Vision Engineer",
        "AI Product Manager",
        "Robotics Engineer",
        "Big Data Analyst",
        "Data Engineer Remote",
        "AI Consultant Jobs",
        "Research Scientist AI",
        "Algorithm Developer",
        "Machine Learning Scientist",
        "AI Ethicist Jobs",
        "Reinforcement Learning Research",
        "Quantum Computing Researcher",
        "Autonomous Systems Engineer",
        "AI Hardware Engineer"
    ]
    search_engines = ['google', 'bing']
    all_urls = []

    logging.debug("Starting main process with search queries.")

    for query in search_queries:
        for engine in search_engines:
            urls = perform_web_search(query, search_engine=engine)
            all_urls.extend(urls)

    logging.debug("All retrieved URLs: %s", all_urls)

    model = UnifiedModel.load_model()
    total_rewards = 0
    num_iterations = 10

    # Training loop with dataset loading in chunks
    dataset = load_dataset("HuggingFaceFW/fineweb", "CC-MAIN-2014-23")
    for i in range(num_iterations):
        # Load a small subset of the dataset (0.1%)
        subset = dataset['train'].shuffle(seed=42).select(range(int(len(dataset['train']) * 0.001)))

        for sample in subset:
            # Visit and scrape the website in the dataset
            content, job_listings, relevant_content, timestamp = extract_job_listings_from_url(sample['url'], sample['text'])

            sentiment_score = calculate_sentiment_score(relevant_content)
            reward = calculate_total_reward(len(content), len(job_listings), sentiment_score)

            text_input = [np.random.rand(1, 1) for _ in range(len(relevant_content.split()))]
            action_output, text_output, generated_text_seq = model.forward(np.random.rand(128, 128), text_input)
            generated_text = ' '.join([str(output) for output in generated_text_seq])

            model.backward(np.array([reward]), text_output, generated_text_seq, 0.001)  # Using a learning rate of 0.001

        # Save model after each iteration
        model.save_model()

    print("\n=========== Total Rewards ===========\n")
    print(total_rewards)

    print("\n=========== Error Summary ===========\n")
    if error_log:
        for error in error_log:
            print(f"Error: {error}")
            print("Possible fix: Verify the website structure and update selectors or ensure the URL is valid.\n")
    else:
        print("No errors encountered during the process.")

if __name__ == "__main__":
    main()
