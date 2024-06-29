HOSTED DEMOS COMING EVENTUALLY! Right now its a flask app.
Models from ollama, download these before running:
```
ollama pull llama3:instruct
ollama pull llava-llama3
ollama pull mxbai-embed-large
```
install this requirements.txt as per usual shiz:
```
pip install -r requirements.txt 
```
it should be as simple as running:
```
python main.py
```

and then going to the url output in your terminal.

ollama is the default, but you can change to groq or openai if you have the credentials set as env variables
```
OPENAI_API_KEY=(OPTIONAL)
GROQ_API_KEY=(OPTIONAL)
DEEPGRAM_API_KEY=(OPTIONAL FOR TTS)
```

This script aims to develop a deep lobe connection to an LLM, end goal is to make it autonomously do tasks until it thinks it cant do anything else then reply to you. The web research feature and its lobes with vision is its strengths in my personal opinion.
each lobe will become its own model trained  to help AURORA specifically.
Thanks for the support in any form!
BELOW THIS LINE IS STILL OUTDATED SORTA:
##############################################################3
# AURORA (Artificial Unified Responsive Optimized Reasoning Agent) - AI Chatbot with Brain-like lobe Functions

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Drlordbasil/AURORA)](https://github.com/Drlordbasil/AURORA/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Drlordbasil/AURORA)](https://github.com/Drlordbasil/AURORA/network)
[![Issues](https://img.shields.io/github/issues/Drlordbasil/AURORA)](https://github.com/Drlordbasil/AURORA/issues)
[![Contributors](https://img.shields.io/github/contributors/Drlordbasil/AURORA)](https://github.com/Drlordbasil/AURORA/graphs/contributors)

## Overview

![image](https://github.com/Drlordbasil/AURORA/assets/126736516/54e72cb3-68a4-4142-8d2a-989aae0333b4)




![image](https://github.com/Drlordbasil/AURORA/assets/126736516/d3e851c2-ebbc-4f27-86cf-93e67d060e66)

**AURORA** is a smart AI chatbot that mimics the brain's different functions to provide insightful and context-aware interactions. It's perfect for research, coding help, sentiment analysis, and more!

## Key Features

- **Brain-like Architecture**: Specialized modules (lobes) for various tasks.
- **Real-time Info**: Uses multiple search engines for up-to-date answers.
- **Sentiment Analysis**: Understands the emotional tone of your inputs.
- **Continuous Learning**: Learns and improves from interactions.
![image](https://github.com/Drlordbasil/AURORA/assets/126736516/cb37aca1-a29e-4f1a-a200-1cab5ba981ac)

## Quick Start

### Prerequisites

- Python 3.12+
- Git

### Installation

1. **Clone the repo**:
    ```bash
    git clone https://github.com/Drlordbasil/AURORA.git
    cd AURORA
    ```

2. **Set up a virtual environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Run AURORA

```bash
python main.py
```

### Example Usage

```python
from brain import Brain

brain = Brain()
response = brain.process_input("Tell me about the latest in AI.", print)
print(response)
```

## Contributing

I welcome contributions! Feel free to fork the project, create a new branch, and submit a pull request.

## License

MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

Questions? Reach out to Anthony Snider at [drlordbasil@gmail.com](mailto:drlordbasil@gmail.com).

