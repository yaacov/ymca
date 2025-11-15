// API Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const clearBtn = document.getElementById('clear-btn');
const summaryBtn = document.getElementById('summary-btn');
const exportBtn = document.getElementById('export-btn');
const loadingModal = document.getElementById('loading-modal');
const enableTools = document.getElementById('enable-tools');
const enablePlanning = document.getElementById('enable-planning');
const refineAnswer = document.getElementById('refine-answer');
const temperatureSlider = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperature-value');
const thinkingTimer = document.getElementById('thinking-timer');

// Timer variables
let timerInterval = null;
let timerStartTime = null;

// Configure marked for markdown rendering
marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try {
                return hljs.highlight(code, { language: lang }).value;
            } catch (err) {}
        }
        return hljs.highlightAuto(code).value;
    }
});

// ==================== Utility Functions ====================

function updateTimer() {
    if (timerStartTime) {
        const elapsed = Math.floor((Date.now() - timerStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        
        if (minutes > 0) {
            thinkingTimer.textContent = `${minutes}m ${seconds}s`;
        } else {
            thinkingTimer.textContent = `${seconds}s`;
        }
    }
}

function showLoading() {
    loadingModal.classList.add('active');
    timerStartTime = Date.now();
    thinkingTimer.textContent = '0s';
    timerInterval = setInterval(updateTimer, 100);
}

function hideLoading() {
    loadingModal.classList.remove('active');
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
    timerStartTime = null;
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function formatTimestamp() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ==================== Message Rendering ====================

function addMessage(role, content, timestamp = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    // Create message header
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    
    let icon, label;
    if (role === 'user') {
        icon = '🧑';
        label = 'You';
    } else if (role === 'assistant') {
        icon = '🤖';
        label = 'Assistant';
    } else {
        icon = '💡';
        label = 'System';
    }
    
    const time = timestamp || formatTimestamp();
    headerDiv.innerHTML = `<span>${icon} ${label}</span><span style="margin-left: auto; font-size: 12px;">${time}</span>`;
    
    // Create message content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Render markdown for assistant and system messages
    if (role === 'assistant' || role === 'system') {
        contentDiv.innerHTML = marked.parse(content);
        // Highlight code blocks
        contentDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    } else {
        // Plain text for user messages (but still allow line breaks)
        contentDiv.textContent = content;
        contentDiv.style.whiteSpace = 'pre-wrap';
    }
    
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    scrollToBottom();
    return messageDiv;
}

function addErrorMessage(error) {
    const errorContent = `**Error:** ${error}`;
    addMessage('system', errorContent);
}

// ==================== API Functions ====================

async function sendMessage(message) {
    if (!message.trim()) return;
    
    // Add user message to chat
    addMessage('user', message);
    
    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';
    
    // Show loading indicator
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                enable_tools: enableTools.checked,
                enable_planning: enablePlanning.checked,
                refine_answer: refineAnswer.checked,
                temperature: parseFloat(temperatureSlider.value),
                max_iterations: 5
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Server error');
        }
        
        const data = await response.json();
        
        // Add assistant response
        addMessage('assistant', data.response);
        
    } catch (error) {
        console.error('Chat error:', error);
        addErrorMessage(error.message);
    } finally {
        hideLoading();
    }
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear the conversation history?')) {
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/clear`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Failed to clear history');
        }
        
        // Clear chat container (keep welcome message)
        const welcomeMessage = chatContainer.querySelector('.system-message');
        chatContainer.innerHTML = '';
        if (welcomeMessage) {
            chatContainer.appendChild(welcomeMessage);
        }
        
        addMessage('system', '✓ Conversation history cleared');
        
    } catch (error) {
        console.error('Clear error:', error);
        addErrorMessage(error.message);
    } finally {
        hideLoading();
    }
}

async function showSummary() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/summary`);
        
        if (!response.ok) {
            throw new Error('Failed to get summary');
        }
        
        const data = await response.json();
        addMessage('system', `**Conversation Summary:**\n\n${data.summary}`);
        
    } catch (error) {
        console.error('Summary error:', error);
        addErrorMessage(error.message);
    } finally {
        hideLoading();
    }
}

async function exportConversation() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/export`);
        
        if (!response.ok) {
            throw new Error('Failed to export conversation');
        }
        
        const data = await response.json();
        
        // Create a downloadable file
        const blob = new Blob([JSON.stringify(data.conversation, null, 2)], { 
            type: 'application/json' 
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `conversation-${new Date().toISOString()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        addMessage('system', `✓ Conversation exported (${data.conversation.length} messages)`);
        
    } catch (error) {
        console.error('Export error:', error);
        addErrorMessage(error.message);
    } finally {
        hideLoading();
    }
}

// ==================== Event Listeners ====================

sendBtn.addEventListener('click', () => {
    const message = userInput.value.trim();
    if (message) {
        sendMessage(message);
    }
});

userInput.addEventListener('keydown', (e) => {
    // Send on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const message = userInput.value.trim();
        if (message) {
            sendMessage(message);
        }
    }
});

// Auto-resize textarea
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = userInput.scrollHeight + 'px';
});

clearBtn.addEventListener('click', clearHistory);
summaryBtn.addEventListener('click', showSummary);
exportBtn.addEventListener('click', exportConversation);

// Update temperature display
temperatureSlider.addEventListener('input', (e) => {
    temperatureValue.textContent = e.target.value;
});

// ==================== Initialization ====================

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();
        
        if (data.status === 'healthy' && data.chat_initialized) {
            console.log('API is healthy and chat is initialized');
        } else {
            addMessage('system', '⚠️ Warning: Chat API may not be fully initialized');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        addErrorMessage('Failed to connect to API server. Please check if the server is running.');
    }
}

// Focus input on load
userInput.focus();

// Check health
checkHealth();

// Add keyboard shortcut hints
console.log('Keyboard shortcuts:');
console.log('  Enter - Send message');
console.log('  Shift+Enter - New line');

