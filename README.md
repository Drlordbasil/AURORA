STABLE: Yes

NOTE: I only use local modals on ollama and groq because its free. Deepgram gave me $200 in credit, or else it wouldnt use voice.
- Thank you :D
# AURORA - Artificial Unified Responsive Optimized Reasoning Agent

AURORA (Artificial Unified Responsive Optimized Reasoning Agent) is a sophisticated AI system designed to simulate the human brain's reasoning process. It utilizes multiple "lobes" that function similarly to human brain lobes, each responsible for different types of cognitive tasks. These lobes work together to provide coherent, insightful, and contextually appropriate responses to user prompts.
![image](https://github.com/Drlordbasil/AURORA/assets/126736516/22d9ea6e-11e4-483a-9d50-6d61180c1031)

## Features

- **Frontal Lobe**: Focuses on reasoning, planning, and problem-solving.
- **Parietal Lobe**: Provides educational insights and processes sensory information.
- **Temporal Lobe**: Contextualizes user prompts socially and processes auditory information.
- **Occipital Lobe**: Describes visual information vividly and clearly.
- **Function Calling**:
  - **Run Local Commands**: Execute local system commands such as listing directory contents, printing text, displaying the current date and time, and checking the Python version installed.
  - **Web Research**: Perform web research queries to gather information from online sources, including performing Google searches, extracting relevant content from top search results, and aggregating the information.
  - **Analyze Images**: Analyze images from provided URLs and generate descriptions of the image content.
  - **Extract Text from PDFs**: Extract text content from PDF files using the PDF URL.
  - **Analyze Sentiment**: Analyze the sentiment of given text, providing polarity and subjectivity scores.
- **Text-to-Speech**: Converts text to speech, with the ability to handle long texts by chunking into smaller parts and combining them into a single playable audio file.
- **Embeddings and Memory Retrieval**:
  - **Generate Embeddings**: Generate embeddings for given text and store them in a collection.
  - **Retrieve Relevant Memory**: Retrieve relevant memories based on prompt embeddings to provide contextually appropriate responses.
- **Central Processing Agent**: Coordinates the entire process of handling user prompts, including running function calls, saving responses to memory, generating embeddings, retrieving memory, processing responses from different lobes, and generating a final coherent response.
- **Error Handling**: Robust error handling with retries and user feedback to ensure reliable operations and informative error messages.
- **GUI Features**:
  - **Info Box**: Displays detailed information about AURORA and its features.
  - **Status Updates**: Provides real-time status updates on the right side of the GUI.
  - **Graceful Exit**: Ensures the application closes all loops and exits gracefully when the window is closed.
  - **Colorful and Professional Design**: A visually appealing and user-friendly interface with a modern and professional look.
  - **Animated Status Bar**: Status bar animations for an engaging user experience.
  - **Logo**: A dedicated space for the AURORA logo to enhance the visual appeal of the GUI.
- **Integration with Advanced AI Models**: Utilizes advanced AI models and APIs such as Groq and Ollama for various functionalities.
- **Voice Output**: Converts final responses to speech for auditory feedback using Deepgram's Text-to-Speech capabilities.


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
- `image_vision.py`: defines ollama vision modal to call as a function in function_calling.py
- `speaker.py`: adds text to speech using deepgram API, can disable in brain.py with commenting out text_to_speech(final_thought) line.
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
