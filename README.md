
# AURORA - Artificial Unified Responsive Optimized Reasoning Agent

AURORA (Artificial Unified Responsive Optimized Reasoning Agent) is a sophisticated AI system designed to simulate the human brain's reasoning process. It utilizes multiple "lobes" that function similarly to human brain lobes, each responsible for different types of cognitive tasks. These lobes work together to provide coherent, insightful, and contextually appropriate responses to user prompts.

## Features

- **Frontal Lobe**: Focuses on reasoning, planning, and problem-solving.
- **Parietal Lobe**: Provides educational insights and processes sensory information.
- **Temporal Lobe**: Contextualizes user prompts socially and processes auditory information.
- **Occipital Lobe**: Describes visual information vividly and clearly.

## Installation

To install the necessary dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Setting Up Environment

Ensure you have set the `GROQ_API_KEY` environment variable:

```bash
export GROQ_API_KEY='your_api_key_here'
```

### Running the Chatbot

To start the chatbot loop, execute the following command:

```bash
python main.py
```

## Files

- `brain.py`: Contains the Brain class, which manages the different lobes and processes user prompts.
- `utilities.py`: Provides utility functions for logging and setting up the embedding collection.
- `final_agent_persona.py`: Defines the persona and user information for the final response agent.
- `function_calling.py`: defines function calling with run_conversation to auto call functions.
- `main.py`: The entry point for running the chatbot loop.

## Contributing

We welcome contributions to enhance AURORA. Please follow the standard GitHub workflow to submit your changes.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License.

## Contact

For any questions or issues, please open an issue on GitHub or contact us directly.

## Acknowledgements

AURORA is built using various technologies and libraries, including Groq, Ollama, and ChromaDB.
