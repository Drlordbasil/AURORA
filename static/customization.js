// DOM Elements
const elements = {
    chatDisplay: document.getElementById('chat-display'),
    promptInput: document.getElementById('prompt-input'),
    sendButton: document.getElementById('send-button'),
    clearButton: document.getElementById('clear-button'),
    ttsToggleButton: document.getElementById('tts-toggle'),
    recordButton: document.getElementById('record-button'),
    statusLabel: document.getElementById('status-label'),
    themeToggle: document.getElementById('theme-toggle'),
    infoPanel: document.querySelector('.info-panel'),
    infoToggle: document.getElementById('info-toggle'),
    artifactDisplay: document.getElementById('artifact-display'),
    apiProviderSelect: document.getElementById('api-provider')
};

// State
const state = {
    isRecording: false,
    isDarkTheme: true,
    activeLobes: new Set(),
    eventSource: null,
    currentApiProvider: 'ollama' // Default API provider
};

// Utility Functions
const utils = {
    formatLongText(text, maxLineLength = 80) {
        return text.split(' ').reduce((lines, word) => {
            if (lines[lines.length - 1].length + word.length + 1 > maxLineLength) {
                lines.push(word);
            } else {
                lines[lines.length - 1] += ' ' + word;
            }
            return lines;
        }, ['']).join('\n');
    },

    async fetchJSON(url, options = {}) {
        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }
};

// UI Update Functions
const ui = {
    displayMessage(message, isUser = true, type = 'normal') {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${isUser ? 'user-message' : 'aurora-message'} ${type}-message`;
        
        message = message.replace(/```([\s\S]*?)```/g, (match, p1) => {
            return `<div class="code-block">${p1}</div>`;
        });

        message = utils.formatLongText(message);

        messageElement.innerHTML = type === 'individual' ? message : `${isUser ? 'You' : 'AURORA'}: ${message}`;
        elements.chatDisplay.appendChild(messageElement);
        elements.chatDisplay.scrollTop = elements.chatDisplay.scrollHeight;
    },

    updateStatus(message, animate = false) {
        console.log("Updating status:", message);
        elements.statusLabel.textContent = message;
        elements.statusLabel.style.animation = animate ? 'pulse 1s infinite' : 'none';
        
        const statusElement = document.createElement('div');
        statusElement.className = 'message system-message';
        statusElement.textContent = message;
        elements.chatDisplay.appendChild(statusElement);
        elements.chatDisplay.scrollTop = elements.chatDisplay.scrollHeight;

        ui.updateBrainVisualization(message);
    },

    showTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        elements.chatDisplay.appendChild(typingIndicator);
        elements.chatDisplay.scrollTop = elements.chatDisplay.scrollHeight;
    },

    hideTypingIndicator() {
        const typingIndicator = elements.chatDisplay.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    },

    updateBrainVisualization(message) {
        const lobes = {
            frontal: ['think', 'plan', 'decide'],
            parietal: ['touch', 'spatial', 'navigation'],
            temporal: ['hear', 'memory', 'language'],
            occipital: ['see', 'visual'],
            cerebellum: ['balance', 'coordination', 'precision']
        };

        const activatedLobes = Object.entries(lobes).filter(([lobe, keywords]) => 
            keywords.some(keyword => message.toLowerCase().includes(keyword))
        ).map(([lobe]) => lobe);

        Object.keys(lobes).forEach(lobe => {
            const element = document.getElementById(lobe + '-lobe');
            if (element) {
                element.style.stroke = activatedLobes.includes(lobe) ? '#00ff9d' : 'var(--aurora-color)';
                element.style.filter = activatedLobes.includes(lobe) ? 'url(#glow)' : 'none';
            }
        });

        setTimeout(() => {
            Object.keys(lobes).forEach(lobe => {
                const element = document.getElementById(lobe + '-lobe');
                if (element) {
                    element.style.stroke = 'var(--aurora-color)';
                    element.style.filter = 'none';
                }
            });
        }, 5000);
    },

    updateThemeColors() {
        const root = document.documentElement;
        const theme = state.isDarkTheme ? {
            bgColor: '#0a0a1e',
            textColor: '#e0e0e0',
            accentColor: '#16213e',
            highlightColor: '#0f3460'
        } : {
            bgColor: '#f0f0f0',
            textColor: '#333333',
            accentColor: '#d0d0d0',
            highlightColor: '#c0c0c0'
        };

        Object.entries(theme).forEach(([key, value]) => {
            root.style.setProperty(`--${key.replace(/[A-Z]/g, letter => `-${letter.toLowerCase()}`)}`, value);
        });
    },

    displayArtifact(artifact) {
        elements.artifactDisplay.innerHTML = '';
        const artifactElement = document.createElement('div');
        artifactElement.className = 'artifact';
        artifactElement.innerHTML = `
            <h3>${artifact.title}</h3>
            <pre><code>${artifact.content}</code></pre>
        `;
        elements.artifactDisplay.appendChild(artifactElement);
    }
};

// API Functions
const api = {
    async sendMessage(message) {
        ui.displayMessage(message, true);
        ui.updateStatus('Processing...', true);
        ui.showTypingIndicator();
        
        try {
            const data = await utils.fetchJSON('/send_message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, apiProvider: state.currentApiProvider })
            });
            
            ui.hideTypingIndicator();
            
            if (data.response) {
                ui.displayMessage(data.response, false);
            }
            
            if (data.audio_file) {
                api.playAudio(data.audio_file);
            }

            if (data.artifact) {
                ui.displayArtifact(data.artifact);
            }

            ui.updateStatus(data.status);
        } catch (error) {
            console.error('Error:', error);
            ui.updateStatus('Error occurred');
            ui.hideTypingIndicator();
        }
    },

    async toggleTTS() {
        try {
            const data = await utils.fetchJSON('/toggle_tts', { method: 'POST' });
            if (data.status === 'Error') {
                console.error('Error toggling TTS:', data.error);
                ui.updateStatus('Error toggling TTS');
            } else {
                ui.updateStatus(data.status);
                elements.ttsToggleButton.classList.toggle('active');
            }
        } catch (error) {
            console.error('Error:', error);
            ui.updateStatus('Error toggling TTS');
        }
    },

    async toggleRecording() {
        if (!state.isRecording) {
            try {
                const data = await utils.fetchJSON('/start_recording', { method: 'POST' });
                state.isRecording = true;
                elements.recordButton.innerHTML = '<i class="fas fa-stop"></i>';
                elements.recordButton.classList.add('active');
                ui.updateStatus(data.status, true);
            } catch (error) {
                console.error('Error:', error);
                ui.updateStatus('Error starting recording');
            }
        } else {
            try {
                ui.updateStatus('Stopping recording...', true);
                const data = await utils.fetchJSON('/stop_recording', { method: 'POST' });
                state.isRecording = false;
                elements.recordButton.innerHTML = '<i class="fas fa-microphone"></i>';
                elements.recordButton.classList.remove('active');
                ui.updateStatus('Processing completed');
                if (data.transcription) {
                    ui.displayMessage(data.transcription, true);
                    if (data.response) {
                        ui.displayMessage(data.response, false);
                    }
                    if (data.audio_file) {
                        api.playAudio(data.audio_file);
                    }
                }
            } catch (error) {
                console.error('Error:', error);
                ui.updateStatus('Error stopping recording');
            }
        }
    },

    playAudio(audioFile) {
        const audio = new Audio(`/get_audio/${audioFile}`);
        audio.play();
    },

    async loadChatHistory() {
        try {
            const history = await utils.fetchJSON('/chat_history.json');
            history.forEach(message => ui.displayMessage(message.text, message.user));
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    },

    handleProgressUpdates() {
        console.log("Setting up SSE connection for progress updates");
        state.eventSource = new EventSource('/progress_updates');
        state.eventSource.onmessage = function(event) {
            console.log("Received SSE message:", event.data);
            const data = JSON.parse(event.data);
            ui.updateStatus(data.message, true);
        };
        state.eventSource.onerror = function(error) {
            console.error('Error in progress updates:', error);
            state.eventSource.close();
        };
    },

    async updateApiProvider(provider) {
        try {
            const data = await utils.fetchJSON('/update_api_provider', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider })
            });
            if (data.status === 'success') {
                state.currentApiProvider = provider;
                ui.updateStatus(`API Provider set to ${provider}`);
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            console.error('Error updating API provider:', error);
            ui.updateStatus('Error updating API provider');
        }
    }
};

// Event Listeners
function setupEventListeners() {
    elements.sendButton.addEventListener('click', () => {
        const message = elements.promptInput.value.trim();
        if (message) {
            api.sendMessage(message);
            elements.promptInput.value = '';
        }
    });

    elements.promptInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            elements.sendButton.click();
        }
    });

    elements.clearButton.addEventListener('click', () => {
        elements.chatDisplay.innerHTML = '';
        ui.updateStatus('Chat cleared');
    });

    elements.ttsToggleButton.addEventListener('click', api.toggleTTS);
    elements.recordButton.addEventListener('click', api.toggleRecording);

    elements.themeToggle.addEventListener('click', () => {
        state.isDarkTheme = !state.isDarkTheme;
        document.body.classList.toggle('light-theme');
        ui.updateThemeColors();
    });

    elements.infoToggle.addEventListener('click', () => {
        elements.infoPanel.classList.toggle('show');
    });

    elements.apiProviderSelect.addEventListener('change', (event) => {
        const selectedProvider = event.target.value;
        api.updateApiProvider(selectedProvider);
    });
}

// Initialization
function initializeApp() {
    console.log("Initializing application");
    setupEventListeners();
    api.loadChatHistory();
    api.handleProgressUpdates();
    ui.updateThemeColors();
    ui.updateStatus('Ready');
}

document.addEventListener('DOMContentLoaded', initializeApp);