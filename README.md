# AURORA - Artificial Unified Responsive Optimized Reasoning Agent

AURORA is a multithought AI with multiple brain lobes that feed into its thoughts as separate thoughts into one entity: AURORA. It aims to simulate the human brain's reasoning process by utilizing separate "lobes" to process user prompts from different perspectives and combine their thoughts to generate comprehensive and insightful responses.

## Features

- Utilizes the Groq API and various other libraries for natural language processing and response generation
- Implements a memory manager using ChromaDB for storing and retrieving relevant conversation history
- Employs separate "lobes" (frontal, parietal, temporal, occipital) to process user prompts based on their specific functions (reasoning, educational insights, social context, visual descriptions)
- Combines the thoughts from all lobes to generate a coherent and contextually relevant response
- Provides logging and monitoring capabilities to track the chatbot's performance and identify potential issues
- Supports error handling and graceful degradation in case of exceptions or API failures

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Groq API key
- Required libraries: `groq`, `ollama`, `chromadb`, `psutil`

### Installation

1. Clone the repository:
   ```
   git clone (https://github.com/Drlordbasil/AURORA).git
   ```

2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

3. Set up the Groq API key:
   - Obtain an API key from Groq
   - Set the `GROQ_API_KEY` environment variable with your API key

### Usage

To start the AURORA chatbot, run the following command:
```
python main.py
```

The chatbot will initialize and prompt you to send a message. You can interact with AURORA by typing your messages and receiving responses. To exit the chatbot, type "exit" or "quit".

## Project Structure

- `brain.py`: Contains the core logic of the chatbot, including the `Brain` class and its methods for processing user prompts, generating responses, and managing memory
- `final_agent_persona.py`: Defines the `FinalAgentPersona` class, which represents the persona of AURORA and contains information about its name, role, description, and user information
- `main.py`: The entry point of the application, responsible for initializing the chatbot and starting the conversation loop
- `memory.py`: Implements the `MemoryManager` class for loading, saving, and retrieving relevant conversation history using ChromaDB
- `utilities.py`: Contains utility functions for setting up logging and initializing the embedding collection

## Future Enhancements

- Improve error handling and provide more informative error messages
- Optimize memory management for scalability and efficiency
- Implement personalization features to adapt responses based on individual user preferences and conversation history
- Add support for multiple users with separate conversation histories and preferences
- Integrate with external knowledge sources or APIs to enhance the chatbot's knowledge and response capabilities

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Groq](https://groq.com/) for providing the API for natural language processing
- [ChromaDB](https://www.trychroma.com/) for the memory management functionality
- [ollama](https://github.com/hazyresearch/ollama) for the language model and embeddings
- [psutil](https://github.com/giampaolo/psutil) for performance monitoring
