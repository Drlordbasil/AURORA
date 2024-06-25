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
let activeLobes = new Set();

function displayMessage(message, isUser = true, type = 'normal') {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${isUser ? 'user-message' : 'aurora-message'} ${type}-message`;
    
    message = message.replace(/```([\s\S]*?)```/g, (match, p1) => {
        return `<div class="code-block">${p1}</div>`;
    });

    message = formatLongText(message);

    if (type === 'individual') {
        messageElement.innerHTML = `${message}`;
    } else {
        messageElement.innerHTML = isUser ? `You: ${message}` : `AURORA: ${message}`;
    }
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
    
    const statusElement = document.createElement('div');
    statusElement.className = 'message system-message';
    statusElement.textContent = message;
    chatDisplay.appendChild(statusElement);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;

    updateBrainVisualization(message);
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
        
        if (data.response) {
            displayMessage(data.response, false);
        }
        
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

function updateBrainVisualization(message) {
    const lobes = ['frontal', 'parietal', 'temporal', 'occipital', 'limbic', 'cerebellar', 'brocas', 'wernickes', 'insular', 'association_areas'];
    const mentionedLobes = lobes.filter(lobe => message.toLowerCase().includes(lobe));
    
    mentionedLobes.forEach(lobe => activeLobes.add(lobe));
    
    lobes.forEach(lobe => {
        const element = document.getElementById(`${lobe}-lobe`);
        if (element) {
            element.style.fill = activeLobes.has(lobe) ? '#00ff9d' : '#333333';
            element.style.filter = activeLobes.has(lobe) ? 'url(#glow)' : 'none';
        }
    });

    setTimeout(() => {
        mentionedLobes.forEach(lobe => activeLobes.delete(lobe));
        updateBrainVisualization('');
    }, 5000);
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
            recordButton.innerHTML = '<i class="fas fa-stop"></i>';
            recordButton.style.backgroundColor = '#DC143C';
            updateStatus(data.status, true);
        } catch (error) {
            console.error('Error:', error);
            updateStatus('Error starting recording');
        }
    } else {
        try {
            updateStatus('Stopping recording...', true);
            const response = await fetch('/stop_recording', { method: 'POST' });
            const data = await response.json();
            isRecording = false;
            recordButton.innerHTML = '<i class="fas fa-microphone"></i>';
            recordButton.style.backgroundColor = '';
            updateStatus('Processing completed');
            if (data.transcription) {
                displayMessage(data.transcription, true);
                if (data.response) {
                    displayMessage(data.response, false);
                }
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
    document.body.style.setProperty('--bg-color', isDarkTheme ? '#0a0a1e' : '#f0f0f0');
    document.body.style.setProperty('--text-color', isDarkTheme ? '#e0e0e0' : '#333333');
    document.body.style.setProperty('--accent-color', isDarkTheme ? '#16213e' : '#d0d0d0');
    document.body.style.setProperty('--highlight-color', isDarkTheme ? '#0f3460' : '#c0c0c0');
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

function handleProgressUpdates() {
    const eventSource = new EventSource('/progress_updates');
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateStatus(data.message, true);
    };
    eventSource.onerror = function(error) {
        console.error('Error in progress updates:', error);
        eventSource.close();
    };
}

loadChatHistory();
handleProgressUpdates();
updateStatus('Ready');