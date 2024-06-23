const chatDisplay = document.getElementById('chat-display');
const promptInput = document.getElementById('prompt-input');
const sendButton = document.getElementById('send-button');
const clearButton = document.getElementById('clear-button');
const ttsToggleButton = document.getElementById('tts-toggle');
const recordButton = document.getElementById('record-button');
const statusLabel = document.getElementById('status-label');
const themeToggle = document.getElementById('theme-toggle');
const infoPanel = document.querySelector('.info-panel');
const infoToggle = document.querySelector('.info-toggle');

let isRecording = false;
let isDarkTheme = true;

function displayMessage(message, isUser = true) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${isUser ? 'user-message' : 'aurora-message'}`;
    
    // Format code blocks
    message = message.replace(/```([\s\S]*?)```/g, (match, p1) => {
        return `<div class="code-block">${p1}</div>`;
    });

    // Auto-format long text
    message = formatLongText(message);

    messageElement.innerHTML = isUser ? `You: ${message}` : `AURORA: ${message}`;
    chatDisplay.appendChild(messageElement);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

function formatLongText(text, maxLineLength = 80) {
    return text.split(' ').reduce((lines, word) => {
        if (lines[lines.length - 1].length + word.length + 1 > maxLineLength) {
            lines.push(word);
        } else {
            lines[lines.length - 1] += ' ' + word;
        }
        return lines;
    }, ['']).join('\n');
}

function updateStatus(message, animate = false) {
    statusLabel.textContent = message;
    statusLabel.style.animation = animate ? 'pulse 1s infinite' : 'none';
}

function playAudio(audioFile) {
    const audio = new Audio(`/get_audio/${audioFile}`);
    audio.play();
}

async function sendMessage(message) {
    displayMessage(message, true);
    updateStatus('Processing...', true);
    showTypingIndicator();
    
    try {
        const response = await fetch('/send_message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message }),
        });
        
        const data = await response.json();
        hideTypingIndicator();
        displayMessage(data.response, false);
        if (data.audio_file) {
            playAudio(data.audio_file);
        }
        updateStatus(data.status);
    } catch (error) {
        console.error('Error:', error);
        updateStatus('Error occurred');
        hideTypingIndicator();
    }
}

function showTypingIndicator() {
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<span></span><span></span><span></span>';
    chatDisplay.appendChild(typingIndicator);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = chatDisplay.querySelector('.typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

sendButton.addEventListener('click', () => {
    const message = promptInput.value.trim();
    if (message) {
        sendMessage(message);
        promptInput.value = '';
    }
});

promptInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendButton.click();
    }
});

clearButton.addEventListener('click', () => {
    chatDisplay.innerHTML = '';
    updateStatus('Chat cleared');
});

ttsToggleButton.addEventListener('click', async () => {
    try {
        const response = await fetch('/toggle_tts', { method: 'POST' });
        const data = await response.json();
        updateStatus(data.status);
    } catch (error) {
        console.error('Error:', error);
        updateStatus('Error toggling TTS');
    }
});

recordButton.addEventListener('click', async () => {
    if (!isRecording) {
        try {
            const response = await fetch('/start_recording', { method: 'POST' });
            const data = await response.json();
            isRecording = true;
            recordButton.textContent = 'Stop Recording';
            recordButton.style.backgroundColor = '#DC143C';
            updateStatus(data.status, true);
        } catch (error) {
            console.error('Error:', error);
            updateStatus('Error starting recording');
        }
    } else {
        try {
            const response = await fetch('/stop_recording', { method: 'POST' });
            const data = await response.json();
            isRecording = false;
            recordButton.textContent = 'Record';
            recordButton.style.backgroundColor = '';
            updateStatus('Recording stopped');
            if (data.transcription) {
                displayMessage(data.transcription, true);
                displayMessage(data.response, false);
                if (data.audio_file) {
                    playAudio(data.audio_file);
                }
            }
        } catch (error) {
            console.error('Error:', error);
            updateStatus('Error stopping recording');
        }
    }
});

themeToggle.addEventListener('click', () => {
    isDarkTheme = !isDarkTheme;
    document.body.style.setProperty('--bg-color', isDarkTheme ? '#1a1a2e' : '#ffffff');
    document.body.style.setProperty('--text-color', isDarkTheme ? '#e0e0e0' : '#333333');
    document.body.style.setProperty('--accent-color', isDarkTheme ? '#16213e' : '#f0f0f0');
    document.body.style.setProperty('--highlight-color', isDarkTheme ? '#0f3460' : '#e0e0e0');
});

infoToggle.addEventListener('click', () => {
    infoPanel.classList.toggle('show');
});

async function loadChatHistory() {
    try {
        const response = await fetch('/chat_history.json');
        const history = await response.json();
        history.forEach(message => displayMessage(message.text, message.user));
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

loadChatHistory();
updateStatus('Ready');  